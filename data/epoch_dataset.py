from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest, resolve_path
from .split import load_split, subject_key


def get_record_paths(manifest_path: str | Path, split_path: str | Path, partition: str, channel_mode: str) -> List[Path]:
    if partition not in {"train", "val", "test"}:
        raise ValueError(partition)
    if channel_mode not in {"single", "multi"}:
        raise ValueError(channel_mode)
    manifest = load_manifest(manifest_path)
    split = load_split(split_path)
    allowed = set(split[partition])
    key_name = f"{channel_mode}_path"
    paths = []
    for r in manifest["records"]:
        if subject_key(r["dataset"], r["subject_id"]) in allowed:
            rel = r.get(key_name)
            if rel:
                paths.append(resolve_path(manifest, rel))
    return sorted(paths)


class EpochDataset(Dataset):
    """In-memory NPZ epoch dataset. Inputs are always returned as (C, T)."""

    def __init__(self, paths: Sequence[str | Path], scaler: bool = False):
        xs, ys = [], []
        for path in paths:
            with np.load(path, allow_pickle=True) as z:
                x = z["x"].astype(np.float32)
                y = z["y"].astype(np.int64)
            if x.ndim == 2:
                x = x[:, None, :]
            if x.ndim != 3:
                raise ValueError(f"Unsupported x shape in {path}: {x.shape}")
            if scaler:
                mean = x.mean(axis=-1, keepdims=True)
                std = x.std(axis=-1, keepdims=True)
                x = (x - mean) / (std + 1e-8)
            xs.append(x)
            ys.append(y)
        if not xs:
            raise RuntimeError("No NPZ records loaded")
        self.x = torch.from_numpy(np.concatenate(xs, axis=0))
        self.y = torch.from_numpy(np.concatenate(ys, axis=0))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
