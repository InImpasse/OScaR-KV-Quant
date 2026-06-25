#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


def pct(x: float | None) -> str:
    return "" if x is None else f"{100.0 * x:.1f}"


def fnum(x: float | None) -> str:
    return "" if x is None else f"{x:.1f}"


def summarize(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    cases = data.get("task_states", {}).get("cases", {})
    completed = [c for c in cases.values() if c.get("status") == "ok"]
    correct = sum(1 for c in completed if c.get("correct"))
    total = len(completed)
    tokens = [c.get("tokens") for c in completed if c.get("tokens") is not None]
    tps = [c.get("tps_gen") for c in completed if c.get("tps_gen") is not None]
    avg_tokens = sum(tokens) / len(tokens) if tokens else None
    avg_tps = sum(tps) / len(tps) if tps else None

    return {
        "dataset": data.get("id", ""),
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "ci_lower": data.get("task_states", {}).get("ci_lower"),
        "ci_upper": data.get("task_states", {}).get("ci_upper"),
        "avg_tokens": avg_tokens,
        "avg_tps": avg_tps,
        "json": str(path),
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        raise SystemExit(2)

    run_dir = Path(sys.argv[1])
    raw_dir = run_dir / "raw"
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        stem = path.stem
        if "_" not in stem:
            continue
        variant, dataset = stem.rsplit("_", 1)
        row = {"variant": variant}
        row.update(summarize(path))
        if dataset != row["dataset"]:
            row["dataset"] = dataset
        rows.append(row)

    order = {
        "baseline_bf16": 0,
        "oscar_turbo2_streamk": 1,
        "turbo2_streamk": 2,
        "oscar_int4": 3,
        "plain_int4": 4,
    }
    dataset_order = {"gpqa": 0, "gsm8k": 1}
    rows.sort(key=lambda r: (order.get(str(r["variant"]), 99), dataset_order.get(str(r["dataset"]), 99)))

    fields = [
        "variant",
        "dataset",
        "total",
        "correct",
        "accuracy",
        "ci_lower",
        "ci_upper",
        "avg_tokens",
        "avg_tps",
        "json",
    ]
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("accuracy", "ci_lower", "ci_upper", "avg_tokens", "avg_tps"):
                if out.get(key) is not None:
                    out[key] = f"{float(out[key]):.6f}"
            writer.writerow({k: out.get(k, "") for k in fields})

    lines = [
        "| variant | dataset | correct/total | accuracy % | 95% CI % | avg tokens | avg gen tok/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = ""
        if row.get("ci_lower") is not None and row.get("ci_upper") is not None:
            ci = f"{pct(float(row['ci_lower']))}-{pct(float(row['ci_upper']))}"
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {row['correct']}/{row['total']} | "
            f"{pct(row.get('accuracy'))} | {ci} | {fnum(row.get('avg_tokens'))} | {fnum(row.get('avg_tps'))} |"
        )
    md_path = run_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
