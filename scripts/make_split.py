#!/usr/bin/env python
import argparse
from _bootstrap import ROOT  # noqa: F401
from preprocessing.common import load_yaml
from data.split import make_subject_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    payload = make_subject_split(
        cfg["manifest"], cfg["output"], datasets=cfg["datasets"], seed=int(cfg["seed"]),
        train_ratio=float(cfg["train_ratio"]), val_ratio=float(cfg["val_ratio"]), test_ratio=float(cfg["test_ratio"]),
        require_both_channels=bool(cfg.get("require_both_channels", True)),
        stratify_by_dataset=bool(cfg.get("stratify_by_dataset", True)),
    )
    print(payload["summary"])


if __name__ == "__main__":
    main()
