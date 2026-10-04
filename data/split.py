from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .manifest import load_manifest


def subject_key(dataset: str, subject_id: str) -> str:
    return f"{dataset}::{subject_id}"


def _can_stratify(labels: Sequence[str], test_size: float) -> bool:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return False
    n = len(labels)
    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def _split(keys: List[str], labels: List[str], test_size: float, seed: int, stratify: bool):
    strat = labels if stratify and _can_stratify(labels, test_size) else None
    a, b = train_test_split(keys, test_size=test_size, random_state=seed, shuffle=True, stratify=strat)
    label_map = dict(zip(keys, labels))
    return sorted(a), sorted(b), [label_map[x] for x in a], [label_map[x] for x in b]


def make_subject_split(
    manifest_path: str | Path,
    output: str | Path,
    *,
    datasets: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    require_both_channels: bool = True,
    stratify_by_dataset: bool = True,
) -> dict:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1")
    manifest = load_manifest(manifest_path)
    units: Dict[str, str] = {}
    for r in manifest["records"]:
        if r["dataset"] not in datasets:
            continue
        if require_both_channels and (not r.get("single_path") or not r.get("multi_path")):
            continue
        key = subject_key(r["dataset"], r["subject_id"])
        units[key] = r["dataset"]
    keys = sorted(units)
    labels = [units[k] for k in keys]
    if len(keys) < 3:
        raise ValueError("Need at least 3 subjects to split")

    holdout = val_ratio + test_ratio
    train, temp, _, temp_labels = _split(keys, labels, holdout, seed, stratify_by_dataset)
    relative_test = test_ratio / holdout
    val, test, _, _ = _split(temp, temp_labels, relative_test, seed + 1, stratify_by_dataset)

    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert set(train) | set(val) | set(test) == set(keys)

    payload = {
        "schema_version": 1,
        "split_type": "subject_wise_fixed",
        "group_key": "dataset::subject_id",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "datasets": list(datasets),
        "require_both_channels": require_both_channels,
        "stratify_by_dataset": stratify_by_dataset,
        "train": train,
        "val": val,
        "test": test,
        "summary": {"n_total_subjects": len(keys), "n_train": len(train), "n_val": len(val), "n_test": len(test)},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
