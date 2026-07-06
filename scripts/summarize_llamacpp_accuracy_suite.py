#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


BENCHMARK_LABEL = {
    "gpqa": ("GPQA", "Score"),
    "gsm8k": ("GSM8K", "Accuracy"),
    "math500": ("MATH500", "Score"),
    "humaneval": ("HumanEval", "Pass@1"),
    "aime2025": ("AIME25", "Score"),
}

VARIANT_LABEL = {
    "baseline_bf16": "BF16",
    "oscar_int4": "OSCAR INT4",
    "plain_int4": "Plain INT4",
}


def summarize(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    cases = data.get("task_states", {}).get("cases", {})
    completed = [c for c in cases.values() if c.get("status") == "ok"]
    correct = sum(1 for c in completed if c.get("correct"))
    total = len(completed)
    return {
        "dataset": data.get("id", ""),
        "total": total,
        "correct": correct,
        "score_pct": (100.0 * correct / total) if total else None,
        "json": str(path),
    }


def fmt_score(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def fmt_delta(value: float | None, base: float | None) -> str:
    if value is None or base is None:
        return ""
    delta = value - base
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} pt"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    rows = []
    by_dataset: dict[str, dict[str, dict[str, object]]] = {}
    for path in sorted((run_dir / "raw").glob("*.json")):
        stem = path.stem
        variant, dataset = stem.rsplit("_", 1)
        if stem.endswith("_math500"):
            variant, dataset = stem.removesuffix("_math500"), "math500"
        if stem.endswith("_humaneval"):
            variant, dataset = stem.removesuffix("_humaneval"), "humaneval"
        if stem.endswith("_aime2025"):
            variant, dataset = stem.removesuffix("_aime2025"), "aime2025"
        row = {"variant": variant, **summarize(path)}
        row["dataset"] = dataset
        rows.append(row)
        by_dataset.setdefault(dataset, {})[variant] = row

    csv_path = run_dir / "summary.csv"
    fields = ["variant", "dataset", "total", "correct", "score_pct", "json"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out.get("score_pct") is not None:
                out["score_pct"] = f"{float(out['score_pct']):.6f}"
            writer.writerow({k: out.get(k, "") for k in fields})

    lines = [
        "| Benchmark | Metric | BF16 | OSCAR INT4 | Delta vs BF16 | Plain INT4 | Delta vs BF16 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gpqa", "gsm8k", "math500", "humaneval", "aime2025"):
        label, metric = BENCHMARK_LABEL[dataset]
        data = by_dataset.get(dataset, {})
        bf16 = data.get("baseline_bf16", {}).get("score_pct")
        oscar = data.get("oscar_int4", {}).get("score_pct")
        plain = data.get("plain_int4", {}).get("score_pct")
        lines.append(
            f"| {label} | {metric} | {fmt_score(bf16)} | {fmt_score(oscar)} | "
            f"{fmt_delta(oscar, bf16)} | {fmt_score(plain)} | {fmt_delta(plain, bf16)} |"
        )
    md_path = run_dir / "accuracy_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
