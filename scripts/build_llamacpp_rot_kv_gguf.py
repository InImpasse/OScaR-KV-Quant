#!/usr/bin/env python3
"""End-to-end llama.cpp OSCAR rotation calibration and GGUF baking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.llamacpp_rot_kv.llama_paths import DEFAULT_DATASET
from scripts.lib.llamacpp_rot_kv.pipeline import PipelineConfig, run_llamacpp_rot_kv_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Base, non-rotated GGUF.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Text file or JSONL calibration prompts.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Calibration working/output directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output -rot-kv GGUF.")
    parser.add_argument("--bin", type=Path, default=None, help="llama-rot-kv-calibrate binary (auto-detect if omitted).")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--dump-token-budget", type=int)
    parser.add_argument("--calib-profile", choices=("smoke", "paper"))
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--predict", type=int, default=1)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--flash-attn", choices=["on", "off", "1", "0"], default="on")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dump-workers", type=int, default=1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--multi-prompt-batch-size", type=int, default=8)
    parser.add_argument("--skip-dump", action="store_true", help="Reuse an existing work-dir/qkv_dumps/llamacpp.")
    parser.add_argument("--skip-compute", action="store_true", help="Reuse existing .pt rotations under work-dir/rotations.")
    parser.add_argument("--resume", action="store_true", help="Resume dump/rotation when stage manifests still match.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack-run", action="store_true", help="Required when actually running llama-rot-kv-calibrate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and not args.skip_dump and not args.ack_run:
        raise SystemExit("refusing to run llama-rot-kv-calibrate without --ack-run; use --dry-run to inspect")

    calibrator_options: dict = {}
    if args.bin is not None:
        calibrator_options["calibrator_bin"] = str(args.bin)

    run_llamacpp_rot_kv_pipeline(
        PipelineConfig(
            base_gguf=args.base,
            dataset=args.dataset,
            work_dir=args.work_dir,
            output_gguf=args.out,
            max_prompts=args.max_prompts,
            dump_token_budget=args.dump_token_budget,
            calib_profile=args.calib_profile,
            ctx=args.ctx,
            predict=args.predict,
            ngl=args.ngl,
            flash_attn=args.flash_attn,
            head_dim=args.head_dim,
            dump_workers=args.dump_workers,
            threads=args.threads,
            multi_prompt_batch_size=args.multi_prompt_batch_size,
            skip_dump=args.skip_dump,
            skip_compute=args.skip_compute,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            calibrator_options=calibrator_options,
        )
    )


if __name__ == "__main__":
    main()
