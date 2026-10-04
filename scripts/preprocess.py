#!/usr/bin/env python
import argparse
from _bootstrap import ROOT  # noqa: F401
from preprocessing.common import load_yaml
from preprocessing import hmc, sleepedf, isruc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/preprocessing.yaml")
    p.add_argument("--dataset", choices=["hmc", "sleepedf", "isruc", "all"], required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    builders = {"hmc": hmc.build, "sleepedf": sleepedf.build, "isruc": isruc.build}
    names = list(builders) if args.dataset == "all" else [args.dataset]
    for name in names:
        builders[name](cfg, limit=args.limit)


if __name__ == "__main__":
    main()
