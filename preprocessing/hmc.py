from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import mne
from tqdm import tqdm

from .common import (
    annotations_to_epoch_labels,
    apply_normalization,
    ensure_dir,
    epoch_raw_with_labels,
    filter_resample_raw,
    resolve_exact_channels,
    resolve_trim_policy,
    save_npz,
    trim_sleep_window,
    validate_xy,
)


def pair_files(root: str | Path) -> List[Tuple[str, Path, Path]]:
    root = Path(root)
    pairs = []
    for signal in sorted(root.glob("SN[0-9][0-9][0-9].edf")):
        record_id = signal.stem
        scoring = root / f"{record_id}_sleepscoring.edf"
        if scoring.exists():
            pairs.append((record_id, signal, scoring))
    return pairs


def _process_record(record_id: str, signal: Path, scoring: Path, mode: str, cfg: dict, global_cfg: dict, out_dir: Path):
    target_fs = int(global_cfg["target_fs"])
    epoch_sec = int(global_cfg["epoch_sec"])
    low_hz = float(global_cfg["filter"]["low_hz"])
    high_hz = float(global_cfg["filter"]["high_hz"])
    normalization = global_cfg.get("normalization", "epoch")
    wanted = cfg[f"{mode}_channels"]

    raw = mne.io.read_raw_edf(signal, preload=True, verbose="ERROR")
    picked = resolve_exact_channels(raw, wanted)
    raw = filter_resample_raw(raw, picked, target_fs, low_hz, high_hz)
    ann = mne.read_annotations(str(scoring))
    total_epochs = raw.n_times // (target_fs * epoch_sec)
    labels = annotations_to_epoch_labels(ann, total_epochs, epoch_sec)
    x, y = epoch_raw_with_labels(raw, labels, target_fs, epoch_sec)
    raw.close()

    trim_enabled, keep = resolve_trim_policy(global_cfg, cfg)
    if trim_enabled:
        x, y = trim_sleep_window(x, y, keep)
    x = apply_normalization(x, normalization)
    validate_xy(x, y, target_fs=target_fs, epoch_sec=epoch_sec)

    prep = {
        "target_fs": target_fs,
        "epoch_sec": epoch_sec,
        "filter_hz": [low_hz, high_hz],
        "normalization": normalization,
        "trim_wake": trim_enabled,
        "keep_wake_epochs_each_side": keep,
        "mode": mode,
    }
    save_npz(
        out_dir / f"HMC__{record_id}.npz",
        x=x,
        y=y,
        fs=target_fs,
        channel_roles=["EEG"] if mode == "single" else ["EEG", "EOG"],
        original_channel_names=picked,
        subject_id=record_id,
        record_id=record_id,
        dataset="HMC",
        preprocessing_config=prep,
        source_signal=str(signal),
        source_label=str(scoring),
    )


def build(global_cfg: dict, limit: int | None = None) -> None:
    cfg = global_cfg["datasets"]["hmc"]
    root = Path(cfg["raw_root"])
    out_root = Path(cfg["out_root"])
    pairs = pair_files(root)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"No HMC pairs found in {root}")

    for mode in ("single", "multi"):
        out_dir = ensure_dir(out_root / f"{mode}_channel")
        for record_id, signal, scoring in tqdm(pairs, desc=f"HMC {mode}"):
            try:
                _process_record(record_id, signal, scoring, mode, cfg, global_cfg, out_dir)
            except Exception as exc:
                print(f"[HMC][{mode}][ERROR] {record_id}: {exc}")
