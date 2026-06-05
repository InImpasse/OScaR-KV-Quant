#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


def as_float(value: str):
    if value is None or value == "":
        return None
    return float(value)


def as_int(value: str, default=0):
    if value is None or value == "":
        return default
    return int(value)


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_key(row):
    return row["model"], row["length"], row["context"]


def main():
    parser = argparse.ArgumentParser(description="Gate KV-cache PPL summary rows against same-context F16 baselines.")
    parser.add_argument("summary_csv", type=Path, help="CSV produced by summarize_kv_ppl.py")
    parser.add_argument("--max-ratio", type=float, default=1.05, help="Maximum allowed PPL ratio vs F16 for non-baseline rows.")
    parser.add_argument("--max-delta", type=float, help="Optional maximum allowed absolute PPL delta vs F16.")
    parser.add_argument("--allow-kv", action="append", default=[], help="Only gate these KV labels; may be repeated.")
    parser.add_argument("--fail-baseline", action="store_true", help="Fail if F16 baseline rows are missing for any row key.")
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    baselines = {row_key(row): row for row in rows if row.get("kv") == "f16"}
    allow = set(args.allow_kv)

    failures = []
    checked = 0
    for row in rows:
        key = row_key(row)
        kv = row.get("kv", "")
        exit_code = as_int(row.get("exit_code"), default=-1)
        ppl = as_float(row.get("ppl"))

        if exit_code != 0:
            failures.append(f"{row.get('label', kv)}: exit_code={exit_code}")
            continue
        if ppl is None:
            failures.append(f"{row.get('label', kv)}: missing PPL")
            continue
        if kv == "f16":
            continue
        if allow and kv not in allow:
            continue

        base = baselines.get(key)
        if base is None:
            if args.fail_baseline:
                failures.append(f"{row.get('label', kv)}: missing F16 baseline for {key}")
            continue

        ratio = as_float(row.get("ppl_ratio_vs_f16"))
        delta = as_float(row.get("ppl_delta_vs_f16"))
        if ratio is None:
            failures.append(f"{row.get('label', kv)}: missing PPL ratio vs F16")
            continue

        checked += 1
        if ratio > args.max_ratio:
            failures.append(
                f"{row.get('label', kv)}: PPL ratio {ratio:.6g} > max {args.max_ratio:.6g}"
            )
        if args.max_delta is not None and delta is not None and delta > args.max_delta:
            failures.append(
                f"{row.get('label', kv)}: PPL delta {delta:.6g} > max {args.max_delta:.6g}"
            )

    if failures:
        print(f"PPL gate failed: checked={checked}, failures={len(failures)}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"PPL gate passed: checked={checked}, max_ratio={args.max_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
