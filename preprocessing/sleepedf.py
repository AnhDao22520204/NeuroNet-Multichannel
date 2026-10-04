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


def pair_files(root: str | Path) -> List[Tuple[str, str, Path, Path]]:
    root = Path(root)
    psg_files = sorted(root.glob("SC*PSG.edf"))
    hyp_files = sorted(root.glob("SC*Hypnogram.edf"))
    hyp_map = {p.name: p for p in hyp_files}
    pairs = []
    for psg in psg_files:
        record_id = psg.name.replace("-PSG.edf", "")  # e.g. SC4001E0
        subject_id = record_id[:6]                      # groups E0/E1 for one participant
        expected = []
        if record_id.endswith("E0"):
            expected.append(f"{subject_id}EC-Hypnogram.edf")
        if record_id.endswith("E1"):
            expected.append(f"{subject_id}EH-Hypnogram.edf")
        hyp = next((hyp_map[n] for n in expected if n in hyp_map), None)
        if hyp is None:
            candidates = [p for p in hyp_files if p.name.startswith(subject_id)]
            if len(candidates) == 1:
                hyp = candidates[0]
        if hyp is not None:
            pairs.append((subject_id, record_id, psg, hyp))
        else:
            print(f"[SleepEDF][WARN] No hypnogram for {psg.name}")
    return pairs


def _process_record(subject_id: str, record_id: str, signal: Path, hyp: Path, mode: str, cfg: dict, global_cfg: dict, out_dir: Path):
    target_fs = int(global_cfg["target_fs"])
    epoch_sec = int(global_cfg["epoch_sec"])
    low_hz = float(global_cfg["filter"]["low_hz"])
    high_hz = float(global_cfg["filter"]["high_hz"])
    normalization = global_cfg.get("normalization", "epoch")
    wanted = cfg[f"{mode}_channels"]

    raw = mne.io.read_raw_edf(signal, preload=True, verbose="ERROR")
    picked = resolve_exact_channels(raw, wanted)
    raw = filter_resample_raw(raw, picked, target_fs, low_hz, high_hz)
    ann = mne.read_annotations(str(hyp))
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
        out_dir / f"SleepEDF__{record_id}.npz",
        x=x,
        y=y,
        fs=target_fs,
        channel_roles=["EEG"] if mode == "single" else ["EEG", "EOG"],
        original_channel_names=picked,
        subject_id=subject_id,
        record_id=record_id,
        dataset="SleepEDF",
        preprocessing_config=prep,
        source_signal=str(signal),
        source_label=str(hyp),
    )


def build(global_cfg: dict, limit: int | None = None) -> None:
    cfg = global_cfg["datasets"]["sleepedf"]
    root = Path(cfg["raw_root"])
    out_root = Path(cfg["out_root"])
    pairs = pair_files(root)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"No Sleep-EDF Sleep Cassette pairs found in {root}")

    for mode in ("single", "multi"):
        out_dir = ensure_dir(out_root / f"{mode}_channel")
        for subject_id, record_id, signal, hyp in tqdm(pairs, desc=f"SleepEDF {mode}"):
            try:
                _process_record(subject_id, record_id, signal, hyp, mode, cfg, global_cfg, out_dir)
            except Exception as exc:
                print(f"[SleepEDF][{mode}][ERROR] {record_id}: {exc}")
