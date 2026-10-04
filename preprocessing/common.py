from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import yaml

LABEL_MAP = {
    "Sleep stage W": 0,
    "W": 0,
    "Sleep stage N1": 1,
    "Sleep stage 1": 1,
    "N1": 1,
    "1": 1,
    "Sleep stage N2": 2,
    "Sleep stage 2": 2,
    "N2": 2,
    "2": 2,
    "Sleep stage N3": 3,
    "Sleep stage N4": 3,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "N3": 3,
    "N4": 3,
    "3": 3,
    "4": 3,
    "Sleep stage R": 4,
    "R": 4,
    "REM": 4,
}

DROP_LABELS = {
    "Sleep stage ?",
    "?",
    "Sleep stage M",
    "Sleep stage movement",
    "Movement time",
    "Movement",
    "Unknown",
    "M",
}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_exact_channels(raw: mne.io.BaseRaw, wanted: Sequence[str]) -> List[str]:
    channel_map = {normalize_channel_name(ch): ch for ch in raw.ch_names}
    resolved = []
    for ch in wanted:
        key = normalize_channel_name(ch)
        if key not in channel_map:
            raise ValueError(f"Missing channel {ch!r}. Available: {raw.ch_names}")
        resolved.append(channel_map[key])
    return resolved


def resolve_candidate_channel(raw: mne.io.BaseRaw, candidates: Sequence[str]) -> str:
    channel_map = {normalize_channel_name(ch): ch for ch in raw.ch_names}
    for candidate in candidates:
        key = normalize_channel_name(candidate)
        if key in channel_map:
            return channel_map[key]
    # Conservative fuzzy fallback for datasets with small naming variations.
    for candidate in candidates:
        key = normalize_channel_name(candidate)
        for norm_name, original in channel_map.items():
            if key in norm_name or norm_name in key:
                return original
    raise ValueError(f"No channel matched candidates={list(candidates)}. Available: {raw.ch_names}")


def filter_resample_raw(
    raw: mne.io.BaseRaw,
    channels: Sequence[str],
    target_fs: int,
    low_hz: float,
    high_hz: float,
) -> mne.io.BaseRaw:
    raw.pick(list(channels))
    sfreq = float(raw.info["sfreq"])
    safe_high = min(float(high_hz), sfreq / 2.0 - 0.5)
    if safe_high <= low_hz:
        raise ValueError(f"Sampling rate too low for filter: fs={sfreq}, low={low_hz}, high={safe_high}")
    raw.filter(l_freq=float(low_hz), h_freq=safe_high, method="fir", verbose="ERROR")
    if int(round(float(raw.info["sfreq"]))) != int(target_fs):
        raw.resample(int(target_fs), verbose="ERROR")
    return raw


def annotations_to_epoch_labels(
    annotations: mne.Annotations,
    total_epochs: int,
    epoch_sec: int,
) -> List[Optional[int]]:
    labels: List[Optional[int]] = [None] * total_epochs
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        desc = str(desc).strip()
        if desc in DROP_LABELS:
            continue
        if desc.startswith(("Lights off", "Lights on", "Recording starts", "Recording ends")):
            continue
        if desc not in LABEL_MAP:
            continue
        label = LABEL_MAP[desc]
        n_epochs = max(1, int(round(float(duration) / epoch_sec)))
        start_epoch = int(round(float(onset) / epoch_sec))
        for offset in range(n_epochs):
            idx = start_epoch + offset
            if 0 <= idx < total_epochs:
                labels[idx] = label
    return labels


def epoch_raw_with_labels(
    raw: mne.io.BaseRaw,
    labels: Sequence[Optional[int]],
    target_fs: int,
    epoch_sec: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = raw.get_data().astype(np.float32)  # (C, T)
    epoch_samples = int(target_fs * epoch_sec)
    usable_epochs = min(data.shape[1] // epoch_samples, len(labels))
    xs, ys = [], []
    for i in range(usable_epochs):
        label = labels[i]
        if label is None:
            continue
        start = i * epoch_samples
        epoch = data[:, start : start + epoch_samples]
        if epoch.shape[-1] != epoch_samples:
            continue
        xs.append(epoch)
        ys.append(int(label))
    if not xs:
        raise ValueError("No valid labeled epochs were produced")
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def trim_sleep_window(
    x: np.ndarray,
    y: np.ndarray,
    keep_wake_epochs_each_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sleep_idx = np.flatnonzero(y != 0)
    if len(sleep_idx) == 0:
        return x, y
    start = max(0, int(sleep_idx[0]) - int(keep_wake_epochs_each_side))
    end = min(len(y), int(sleep_idx[-1]) + int(keep_wake_epochs_each_side) + 1)
    return x[start:end], y[start:end]


def normalize_epoch_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def normalize_subject_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def apply_normalization(x: np.ndarray, method: str) -> np.ndarray:
    method = str(method).lower()
    if method == "epoch":
        return normalize_epoch_channel(x)
    if method == "subject":
        return normalize_subject_channel(x)
    if method == "none":
        return x.astype(np.float32)
    raise ValueError("normalization must be one of: epoch, subject, none")


def resolve_trim_policy(global_cfg: dict, dataset_cfg: dict) -> Tuple[bool, int]:
    if "trim_wake" in dataset_cfg:
        enabled = bool(dataset_cfg["trim_wake"])
        keep = int(dataset_cfg.get("keep_epochs_each_side", 60))
        return enabled, keep
    wake_cfg = global_cfg.get("wake_trim", {})
    return bool(wake_cfg.get("enabled", False)), int(wake_cfg.get("keep_epochs_each_side", 60))


def save_npz(
    out_path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_roles: Sequence[str],
    original_channel_names: Sequence[str],
    subject_id: str,
    record_id: str,
    dataset: str,
    preprocessing_config: dict,
    source_signal: Optional[str] = None,
    source_label: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    payload = {
        "x": x.astype(np.float32),
        "y": y.astype(np.int64),
        "fs": np.asarray(fs, dtype=np.int32),
        "ch_names": np.asarray(list(channel_roles)),
        "original_ch_names": np.asarray(list(original_channel_names)),
        "subject_id": np.asarray(str(subject_id)),
        "record_id": np.asarray(str(record_id)),
        "dataset": np.asarray(str(dataset)),
        "preprocessing_config": np.asarray(json.dumps(preprocessing_config, ensure_ascii=False, sort_keys=True)),
    }
    if source_signal is not None:
        payload["source_signal"] = np.asarray(str(source_signal))
    if source_label is not None:
        payload["source_label"] = np.asarray(str(source_label))
    np.savez_compressed(out_path, **payload)


def validate_xy(x: np.ndarray, y: np.ndarray, *, target_fs: int, epoch_sec: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"Expected x=(N,C,T), got {x.shape}")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError(f"Invalid y shape {y.shape} for x {x.shape}")
    expected = int(target_fs * epoch_sec)
    if x.shape[-1] != expected:
        raise ValueError(f"Expected T={expected}, got {x.shape[-1]}")
    if not set(np.unique(y).tolist()).issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Labels outside 0..4: {np.unique(y)}")
