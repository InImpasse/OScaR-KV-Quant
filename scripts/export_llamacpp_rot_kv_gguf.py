#!/usr/bin/env python3
"""Bake OSCAR K/V rotation checkpoints into a llama.cpp GGUF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.llamacpp_rot_kv.bake_gguf import (
    DEFAULT_K_ROT,
    DEFAULT_V_ROT,
    BakeGgufConfig,
    bake_rotations_to_gguf,
    derive_output_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Base GGUF to copy.")
    parser.add_argument("--rot-dir", type=Path, required=True, help=f"Directory containing {DEFAULT_K_ROT} and {DEFAULT_V_ROT}.")
    parser.add_argument("--out", type=Path, help="Output GGUF. Defaults to BASE stem + -rot-kv.gguf.")
    parser.add_argument("--k-rotation-filename", default=DEFAULT_K_ROT)
    parser.add_argument("--v-rotation-filename", default=DEFAULT_V_ROT)
    parser.add_argument("--max-orthogonality-error", type=float, default=1e-4)
    parser.add_argument("--allow-layer-mismatch", action="store_true")
    parser.add_argument("--replace-rotations", action="store_true", help="Replace existing rotation tensors if present.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print what would be written.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bake_rotations_to_gguf(
        BakeGgufConfig(
            base=args.base,
            rot_dir=args.rot_dir,
            out_path=args.out or derive_output_path(args.base),
            k_rotation_filename=args.k_rotation_filename,
            v_rotation_filename=args.v_rotation_filename,
            max_orthogonality_error=args.max_orthogonality_error,
            allow_layer_mismatch=args.allow_layer_mismatch,
            replace_rotations=args.replace_rotations,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
