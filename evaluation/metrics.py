from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABEL_IDS = [0, 1, 2, 3, 4]
LABEL_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_metrics(y_true, y_pred) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "n_epochs": int(len(y_true)),
    }
    rows = []
    for name in LABEL_NAMES:
        rows.append({
            "class": name,
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1_score": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        })
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    return summary, pd.DataFrame(rows), cm


def save_metrics(output_dir: str | Path, prefix: str, y_true, y_pred, extra_summary: Dict | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class, cm = compute_metrics(y_true, y_pred)
    if extra_summary:
        summary = {**extra_summary, **summary}
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    per_class.to_csv(output_dir / f"{prefix}_per_class.csv", index=False)
    pd.DataFrame(cm, index=[f"True_{n}" for n in LABEL_NAMES], columns=[f"Pred_{n}" for n in LABEL_NAMES]).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv"
    )
    np.save(output_dir / f"{prefix}_y_true.npy", np.asarray(y_true, dtype=np.int64))
    np.save(output_dir / f"{prefix}_y_pred.npy", np.asarray(y_pred, dtype=np.int64))
    return summary
