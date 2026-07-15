#!/usr/bin/env python3
"""Export calibration prompts into one JSONL file for rotation fitting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "third_party" / "OSCAR" / "examples" / "llama-eval" / "llama-eval.py"


def load_eval_module(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing llama-eval module: {path}")
    spec = importlib.util.spec_from_file_location("llama_eval_for_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pick_indices(n_total: int, n_cases: int, seed: int) -> list[int]:
    n = min(n_total, n_cases)
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[:n])


def add_gpqa(mod, rows: list[dict], n_cases: int, seed: int) -> None:
    dataset = mod.GpqaDataset(variant="diamond", seed=seed)
    for idx in pick_indices(len(dataset), n_cases, seed):
        question = dataset.get_question(idx)
        rows.append({
            "dataset": "gpqa",
            "index": idx,
            "prompt": dataset.get_prompt(question),
            "expected": dataset.get_answer(question),
        })


def add_gsm8k(mod, rows: list[dict], n_cases: int, seed: int) -> None:
    dataset = mod.Gsm8kDataset()
    for idx in pick_indices(len(dataset), n_cases, seed):
        question = dataset.get_question(idx)
        rows.append({
            "dataset": "gsm8k",
            "index": idx,
            "prompt": dataset.get_prompt(question),
            "expected": dataset.get_answer(question),
        })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "calibration_prompts_gpqa_gsm8k.jsonl")
    parser.add_argument("--llama-eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--datasets", default="gpqa,gsm8k", help="Comma list: gpqa,gsm8k")
    parser.add_argument("--gpqa-n-cases", type=int, default=198)
    parser.add_argument("--gsm8k-n-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the combined output rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mod = load_eval_module(args.llama_eval)

    rows: list[dict] = []
    selected = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    for name in selected:
        if name == "gpqa":
            add_gpqa(mod, rows, args.gpqa_n_cases, args.seed)
        elif name == "gsm8k":
            add_gsm8k(mod, rows, args.gsm8k_n_cases, args.seed)
        else:
            raise SystemExit(f"unsupported dataset: {name}")

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1

    if args.dry_run:
        print(f"would write {args.out}: total={len(rows)} counts={counts}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps({
            "format_version": 1,
            "source": "third_party/OSCAR/examples/llama-eval",
            "datasets": selected,
            "counts": counts,
            "seed": args.seed,
            "rows": len(rows),
            "output": str(args.out),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}: total={len(rows)} counts={counts}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
