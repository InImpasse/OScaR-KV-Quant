"""Summarize accuracy eval directories for BF16 / INT2 / OSCAR INT2 comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_result(path: Path) -> dict:
    metrics = path / "metrics.json"
    if metrics.exists():
        data = json.loads(metrics.read_text())
        return {
            "task": data.get("task", path.parent.name),
            "accuracy": float(data["score"]),
            "num_requests": int(data.get("num_examples", 0)),
            "correct": data.get("correct"),
            "answer_rate": data.get("answer_rate"),
        }
    result = path / "result.jsonl"
    if result.exists():
        lines = [line for line in result.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"empty {result}")
        data = json.loads(lines[-1])
        return {
            "task": data.get("task", path.parent.name),
            "accuracy": float(data["accuracy"]),
            "num_requests": int(data.get("num_requests", 0)),
            "correct": None,
            "answer_rate": None,
        }
    raise FileNotFoundError(f"no metrics.json or result.jsonl under {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--int2", type=Path, required=True)
    parser.add_argument("--oscar-int2", dest="oscar", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = {
        "bf16": _load_result(args.bf16),
        "int2": _load_result(args.int2),
        "oscar-int2": _load_result(args.oscar),
    }
    bf16 = rows["bf16"]["accuracy"]
    for row in rows.values():
        row["delta_vs_bf16"] = row["accuracy"] - bf16
    text = json.dumps(rows, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print("mode        task       n     accuracy  delta_vs_bf16")
    for mode, row in rows.items():
        print(
            f"{mode:<11} {str(row['task']):<10} {row['num_requests']:<5} "
            f"{row['accuracy']:.4f}    {row['delta_vs_bf16']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
