#!/usr/bin/env python3
"""Compute OSCAR K/V rotation .pt files from llama.cpp QKV dumps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.llamacpp_rot_kv.compute_rotation import ComputeRotationConfig, compute_rotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-id", default="all")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--method", choices=["qqt_sst"], default="qqt_sst")
    parser.add_argument(
        "--composition",
        choices=["plain", "pbr", "br", "br_h128", "r_h", "h_pbr", "h_r_pbr", "h_pbr_r", "r_h_pbr"],
        default="r_h_pbr",
    )
    parser.add_argument("--calibration-meta", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    compute_rotations(
        ComputeRotationConfig(
            dump_path=args.dump_path,
            output_dir=args.output_dir,
            head_dim=args.head_dim,
            chunk_id=args.chunk_id,
            method=args.method,
            composition=args.composition,
            calibration_meta=args.calibration_meta,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
