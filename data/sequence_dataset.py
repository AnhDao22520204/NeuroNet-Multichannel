from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Build windows per recording. Evaluation returns record/epoch indices so overlapping
    tail windows can be aggregated and every real epoch contributes only once to metrics.
    """

    def __init__(self, paths: Sequence[str | Path], context_length: int = 20, stride: int = 10):
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.records = []
        self.samples = []
        for record_idx, path in enumerate(paths):
            with np.load(path, allow_pickle=True) as z:
                x = z["x"].astype(np.float32)
                y = z["y"].astype(np.int64)
            if x.ndim == 2:
                x = x[:, None, :]
            if len(x) != len(y):
                raise ValueError(f"x/y mismatch in {path}")
            self.records.append((x, y, str(path)))
            n = len(y)
            if n < self.context_length:
                continue
            starts = list(range(0, n - self.context_length + 1, self.stride))
            tail = n - self.context_length
            if not starts or starts[-1] != tail:
                starts.append(tail)
            for start in starts:
                self.samples.append((record_idx, start))
        if not self.samples:
            raise RuntimeError("No temporal sequences could be built")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_idx, start = self.samples[idx]
        x, y, _ = self.records[record_idx]
        end = start + self.context_length
        epoch_indices = np.arange(start, end, dtype=np.int64)
        return (
            torch.from_numpy(x[start:end]),
            torch.from_numpy(y[start:end]),
            torch.tensor(record_idx, dtype=torch.long),
            torch.from_numpy(epoch_indices),
        )
