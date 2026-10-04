from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np


def aggregate_sequence_predictions(logits_batches, labels_batches, record_batches, index_batches):
    """Average logits for duplicate (record, epoch) appearances, then predict once."""
    sums: Dict[tuple, np.ndarray] = {}
    counts = defaultdict(int)
    truths = {}
    for logits, labels, records, indices in zip(logits_batches, labels_batches, record_batches, index_batches):
        for b in range(logits.shape[0]):
            rec = int(records[b])
            for t in range(logits.shape[1]):
                idx = int(indices[b, t])
                key = (rec, idx)
                sums[key] = sums.get(key, np.zeros(logits.shape[-1], dtype=np.float64)) + logits[b, t]
                counts[key] += 1
                y = int(labels[b, t])
                if key in truths and truths[key] != y:
                    raise ValueError(f"Conflicting labels for {key}")
                truths[key] = y
    keys = sorted(sums)
    y_true = np.asarray([truths[k] for k in keys], dtype=np.int64)
    y_pred = np.asarray([np.argmax(sums[k] / counts[k]) for k in keys], dtype=np.int64)
    return y_true, y_pred
