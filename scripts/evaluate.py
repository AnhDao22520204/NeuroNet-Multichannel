#!/usr/bin/env python
import argparse
from _bootstrap import ROOT  # noqa: F401
from evaluation.evaluate import run_from_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    run_from_config(args.config)
