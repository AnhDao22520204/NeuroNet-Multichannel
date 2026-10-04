from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    apply_normalization,
    ensure_dir,
    filter_resample_raw,
    resolve_candidate_channel,
    resolve_trim_policy,
    save_npz,
    trim_sleep_window,
    validate_xy,
)

TEXT_LABEL_MAP = {
    "W": 0,
    "WAKE": 0,
    "N1": 1,
    "S1": 1,
    "N2": 2,
    "S2": 2,
    "N3": 3,
    "S3": 3,
    "N4": 3,
    "S4": 3,
    "R": 4,
    "REM": 4,
}


def find_signal_file(subject_dir: Path) -> Path:
    files = [p for p in subject_dir.iterdir() if p.is_file()]
    signal_files = [p for p in files if p.suffix.lower() in {".rec", ".edf", ".bdf"}]
    if signal_files:
        return sorted(signal_files, key=lambda p: p.stat().st_size, reverse=True)[0]
    candidates = [p for p in files if p.suffix.lower() not in {".txt", ".xls", ".xlsx", ".csv"}]
    if not candidates:
        raise FileNotFoundError(f"No signal file in {subject_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]


def find_label_file(subject_dir: Path, expert: int = 1) -> Path:
    files = [p for p in subject_dir.iterdir() if p.is_file()]
    tag = f"_{expert}"
    txt = [p for p in files if p.suffix.lower() == ".txt"]
    preferred = [p for p in txt if tag in p.stem]
    if preferred:
        return sorted(preferred)[0]
    if txt:
        return sorted(txt)[0]
    excel = [p for p in files if p.suffix.lower() in {".xls", ".xlsx"} and tag in p.stem]
    if excel:
        return sorted(excel)[0]
    raise FileNotFoundError(f"No ISRUC label file in {subject_dir}")


@contextmanager
def mne_readable_signal(path: Path) -> Iterator[Path]:
    """Yield an EDF/BDF-readable path and remove only temporary links/copies we create."""
    if path.suffix.lower() in {".edf", ".bdf"}:
        yield path
        return
    tmp = path.with_name(path.stem + ".__neuronet_tmp__.edf")
    created = False
    try:
        if not tmp.exists():
            try:
                os.link(path, tmp)
            except OSError:
                shutil.copy2(path, tmp)
            created = True
        yield tmp
    finally:
        if created and tmp.exists():
            tmp.unlink()


def _parse_scalar(value, scheme: str) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {"", "?", "M", "MT", "MOVEMENT", "UNKNOWN"}:
        return None
    if s in TEXT_LABEL_MAP:
        return TEXT_LABEL_MAP[s]
    try:
        raw = int(float(s))
    except ValueError:
        return None

    # Deterministic mapping: never infer semantics from a subject's unique labels.
    if scheme == "rk_0_5":
        # 0=W, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}.get(raw)
    if scheme == "aasm_0_4":
        # 0=W, 1=N1, 2=N2, 3=N3, 4=REM
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(raw)
    if scheme == "rk_1_5":
        # 1=W, 2=N1, 3=N2, 4=N3/N4, 5=REM
        return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}.get(raw)
    raise ValueError(f"Unsupported ISRUC label_scheme={scheme!r}")


def read_labels(path: Path, scheme: str) -> np.ndarray:
    labels: List[int] = []
    if path.suffix.lower() == ".txt":
        for line in path.read_text(errors="ignore", encoding="utf-8").splitlines():
            tokens = re.split(r"[\s,;]+", line.strip())
            for token in reversed(tokens):
                mapped = _parse_scalar(token, scheme)
                if mapped is not None:
                    labels.append(mapped)
                    break
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=None)
        for _, row in df.iterrows():
            for value in reversed(row.tolist()):
                mapped = _parse_scalar(value, scheme)
                if mapped is not None:
                    labels.append(mapped)
                    break
    else:
        raise ValueError(f"Unsupported ISRUC label file: {path}")
    if not labels:
        raise ValueError(f"No labels parsed from {path}")
    return np.asarray(labels, dtype=np.int64)


def epoch_signal(data: np.ndarray, labels: np.ndarray, target_fs: int, epoch_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    length = int(target_fs * epoch_sec)
    n = min(data.shape[1] // length, len(labels))
    xs, ys = [], []
    for i in range(n):
        y = int(labels[i])
        if y not in {0, 1, 2, 3, 4}:
            continue
        epoch = data[:, i * length : (i + 1) * length]
        if epoch.shape[-1] == length:
            xs.append(epoch)
            ys.append(y)
    if not xs:
        raise ValueError("No valid ISRUC epochs")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _load_signal(signal: Path, mode: str, cfg: dict, global_cfg: dict):
    target_fs = int(global_cfg["target_fs"])
    low_hz = float(global_cfg["filter"]["low_hz"])
    high_hz = float(global_cfg["filter"]["high_hz"])
    with mne_readable_signal(signal) as readable:
        raw = mne.io.read_raw_edf(str(readable), preload=True, verbose="ERROR")
        eeg = resolve_candidate_channel(raw, cfg["eeg_candidates"])
        if mode == "single":
            picked = [eeg]
        else:
            eog = resolve_candidate_channel(raw, cfg["eog_candidates"])
            picked = [eeg, eog]
        original_sfreq = float(raw.info["sfreq"])
        processed = filter_resample_raw(raw, picked, target_fs, low_hz, high_hz)
        data = processed.get_data().astype(np.float32)
        processed.close()
        raw.close()
    return data, picked, original_sfreq


def _process_subject(subject_dir: Path, mode: str, cfg: dict, global_cfg: dict, out_dir: Path) -> dict:
    subject_id = f"S{int(subject_dir.name):03d}" if subject_dir.name.isdigit() else subject_dir.name
    record_id = subject_id
    signal = find_signal_file(subject_dir)
    label_file = find_label_file(subject_dir, int(cfg.get("expert", 1)))
    labels = read_labels(label_file, str(cfg.get("label_scheme", "rk_0_5")))

    target_fs = int(global_cfg["target_fs"])
    epoch_sec = int(global_cfg["epoch_sec"])
    normalization = global_cfg.get("normalization", "epoch")
    data, picked, original_sfreq = _load_signal(signal, mode, cfg, global_cfg)
    x, y = epoch_signal(data, labels, target_fs, epoch_sec)
    trim_enabled, keep = resolve_trim_policy(global_cfg, cfg)
    if trim_enabled:
        x, y = trim_sleep_window(x, y, keep)
    x = apply_normalization(x, normalization)
    validate_xy(x, y, target_fs=target_fs, epoch_sec=epoch_sec)

    prep = {
        "target_fs": target_fs,
        "epoch_sec": epoch_sec,
        "filter_hz": [float(global_cfg["filter"]["low_hz"]), float(global_cfg["filter"]["high_hz"])],
        "normalization": normalization,
        "trim_wake": trim_enabled,
        "keep_wake_epochs_each_side": keep,
        "label_scheme": cfg.get("label_scheme", "rk_0_5"),
        "mode": mode,
        "original_sfreq": original_sfreq,
    }
    save_npz(
        out_dir / f"ISRUC__{record_id}.npz",
        x=x,
        y=y,
        fs=target_fs,
        channel_roles=["EEG"] if mode == "single" else ["EEG", "EOG"],
        original_channel_names=picked,
        subject_id=subject_id,
        record_id=record_id,
        dataset="ISRUC",
        preprocessing_config=prep,
        source_signal=str(signal),
        source_label=str(label_file),
    )
    return {
        "subject_id": subject_id,
        "mode": mode,
        "n_epochs": len(y),
        "picked_channels": ";".join(picked),
        "status": "ok",
    }


def build(global_cfg: dict, limit: int | None = None) -> None:
    cfg = global_cfg["datasets"]["isruc"]
    root = Path(cfg["raw_root"])
    out_root = Path(cfg["out_root"])
    subjects = [p for p in root.iterdir() if p.is_dir()]
    subjects = sorted(subjects, key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    if limit is not None:
        subjects = subjects[:limit]
    if not subjects:
        raise RuntimeError(f"No ISRUC subject directories in {root}")

    rows = []
    for mode in ("single", "multi"):
        out_dir = ensure_dir(out_root / f"{mode}_channel")
        for subject in tqdm(subjects, desc=f"ISRUC {mode}"):
            try:
                rows.append(_process_subject(subject, mode, cfg, global_cfg, out_dir))
            except Exception as exc:
                print(f"[ISRUC][{mode}][ERROR] {subject.name}: {exc}")
                rows.append({"subject_id": subject.name, "mode": mode, "n_epochs": 0, "picked_channels": "", "status": f"failed: {exc}"})
    pd.DataFrame(rows).to_csv(out_root / "preprocess_summary.csv", index=False, encoding="utf-8-sig")
