#!/usr/bin/env python3
"""Dump llama.cpp post-RoPE Q/K/V tensors into OSCAR rotation layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.llamacpp_rot_kv.dump_qkv import DumpQkvConfig, dump_qkv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Base, non-rotated GGUF.")
    parser.add_argument("--dataset", type=Path, required=True, help="Text file or JSONL prompts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Calibration output directory.")
    parser.add_argument("--bin", type=Path, default=None, help="llama-rot-kv-calibrate binary.")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--dump-token-budget", type=int, default=None)
    parser.add_argument("--calib-profile", choices=("smoke", "paper"), default=None)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--predict", type=int, default=1)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--flash-attn", choices=["on", "off", "1", "0"], default="on")
    parser.add_argument("--cache-type-k", default="bf16")
    parser.add_argument("--cache-type-v", default="bf16")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--dump-workers", type=int, default=1)
    parser.add_argument("--multi-prompt-batch-size", type=int, default=8)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--keep-raw-dumps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack-run", action="store_true", help="Required when not using --dry-run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and not args.ack_run:
        raise SystemExit("refusing to run llama-rot-kv-calibrate without --ack-run; use --dry-run to inspect")

    options: dict = {}
    if args.bin is not None:
        options["calibrator_bin"] = str(args.bin)

    dump_qkv(
        DumpQkvConfig(
            model=args.model,
            dataset=args.dataset,
            out_dir=args.out_dir,
            options=options,
            max_prompts=args.max_prompts,
            dump_token_budget=args.dump_token_budget,
            calib_profile=args.calib_profile,
            ctx=args.ctx,
            predict=args.predict,
            ngl=args.ngl,
            flash_attn=args.flash_attn,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            threads=args.threads,
            threads_batch=args.threads_batch,
            dump_workers=args.dump_workers,
            multi_prompt_batch_size=args.multi_prompt_batch_size,
            resume_partial=args.resume_partial,
            keep_raw_dumps=args.keep_raw_dumps,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
