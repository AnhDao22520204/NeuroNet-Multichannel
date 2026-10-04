from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

REQUIRED_KEYS = {"x", "y", "fs", "ch_names", "subject_id", "record_id", "dataset"}


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def inspect_npz(path: str | Path) -> Dict:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        missing = REQUIRED_KEYS.difference(z.files)
        if missing:
            raise ValueError(f"{path} missing NPZ keys: {sorted(missing)}")
        x = z["x"]
        y = z["y"]
        if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
            raise ValueError(f"Invalid x/y shapes in {path}: x={x.shape}, y={y.shape}")
        return {
            "dataset": scalar_str(z["dataset"]),
            "subject_id": scalar_str(z["subject_id"]),
            "record_id": scalar_str(z["record_id"]),
            "fs": int(np.asarray(z["fs"]).item()),
            "n_epochs": int(len(y)),
            "n_channels": int(x.shape[1]),
            "n_samples": int(x.shape[2]),
            "labels": sorted(set(y.astype(int).tolist())),
        }
