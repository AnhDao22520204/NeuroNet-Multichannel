from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .schema import inspect_npz


def build_manifest(processed_roots: Iterable[str | Path], output: str | Path, strict_pairing: bool = True) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: Dict[tuple, dict] = {}

    for root in map(Path, processed_roots):
        for mode, expected_channels in (("single", 1), ("multi", 2)):
            folder = root / f"{mode}_channel"
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.npz")):
                meta = inspect_npz(path)
                if meta["n_channels"] != expected_channels:
                    raise ValueError(f"{path}: expected {expected_channels} channels for {mode}, got {meta['n_channels']}")
                key = (meta["dataset"], meta["record_id"])
                entry = records.setdefault(
                    key,
                    {
                        "dataset": meta["dataset"],
                        "subject_id": meta["subject_id"],
                        "record_id": meta["record_id"],
                        "single_path": None,
                        "multi_path": None,
                        "n_epochs": None,
                    },
                )
                if entry["subject_id"] != meta["subject_id"]:
                    raise ValueError(f"Subject mismatch for {key}")
                rel = os.path.relpath(path.resolve(), output.parent.resolve())
                entry[f"{mode}_path"] = Path(rel).as_posix()
                if entry["n_epochs"] is None:
                    entry["n_epochs"] = meta["n_epochs"]
                elif entry["n_epochs"] != meta["n_epochs"]:
                    raise ValueError(f"Single/multi epoch count mismatch for {key}")

    out_records = sorted(records.values(), key=lambda r: (r["dataset"], r["subject_id"], r["record_id"]))
    if strict_pairing:
        missing = [r for r in out_records if not r["single_path"] or not r["multi_path"]]
        if missing:
            example = missing[:5]
            raise ValueError(f"Records missing single or multi pair ({len(missing)} total): {example}")

        # Scientific consistency check: the single EEG epochs and labels must be the
        # exact EEG/labels used by the corresponding multi EEG+EOG record.
        for r in out_records:
            single_path = (output.parent / r["single_path"]).resolve()
            multi_path = (output.parent / r["multi_path"]).resolve()
            with np.load(single_path, allow_pickle=True) as zs, np.load(multi_path, allow_pickle=True) as zm:
                ys, ym = zs["y"], zm["y"]
                xs, xm = zs["x"], zm["x"]
                if not np.array_equal(ys, ym):
                    raise ValueError(f"Single/multi labels differ for {r['dataset']}::{r['record_id']}")
                if xs.shape[0] != xm.shape[0] or xs.shape[-1] != xm.shape[-1]:
                    raise ValueError(f"Single/multi shapes differ for {r['dataset']}::{r['record_id']}")
                if not np.allclose(xs[:, 0, :], xm[:, 0, :], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Single EEG is not identical to multi EEG for {r['dataset']}::{r['record_id']}")

    payload = {
        "schema_version": 1,
        "records": out_records,
        "summary": {
            "n_records": len(out_records),
            "n_subjects": len({(r['dataset'], r['subject_id']) for r in out_records}),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_manifest_path"] = str(path.resolve())
    return payload


def resolve_path(manifest: dict, rel_path: str) -> Path:
    base = Path(manifest["_manifest_path"]).parent
    return (base / rel_path).resolve()
