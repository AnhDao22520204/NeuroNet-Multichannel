#!/usr/bin/env python
import argparse
from _bootstrap import ROOT  # noqa: F401
from data.manifest import build_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-roots", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-missing-pairs", action="store_true")
    args = p.parse_args()
    payload = build_manifest(args.processed_roots, args.output, strict_pairing=not args.allow_missing_pairs)
    print(payload["summary"])


if __name__ == "__main__":
    main()
