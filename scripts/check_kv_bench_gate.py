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


def matches(row, models, lengths, kvs):
    if models and row.get("model") not in models:
        return False
    if lengths and row.get("length") not in lengths:
        return False
    if kvs and row.get("kv") not in kvs:
        return False
    return True


def check_min(row, field, threshold, failures):
    if threshold is None:
        return
    value = as_float(row.get(field))
    label = row.get("label", row.get("kv", "<unknown>"))
    if value is None:
        failures.append(f"{label}: missing {field}")
    elif value < threshold:
        failures.append(f"{label}: {field} {value:.6g} < min {threshold:.6g}")


def check_max(row, field, threshold, failures):
    if threshold is None:
        return
    value = as_float(row.get(field))
    label = row.get("label", row.get("kv", "<unknown>"))
    if value is None:
        failures.append(f"{label}: missing {field}")
    elif value > threshold:
        failures.append(f"{label}: {field} {value:.6g} > max {threshold:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Gate llama-bench KV matrix summaries for memory and speed targets.")
    parser.add_argument("summary_csv", type=Path, help="CSV produced by summarize_kv_matrix.py")
    parser.add_argument("--model", action="append", default=[], help="Only gate this model; may be repeated.")
    parser.add_argument("--length", action="append", default=[], help="Only gate this length; may be repeated.")
    parser.add_argument("--kv", action="append", default=[], help="Only gate this KV label; may be repeated.")
    parser.add_argument("--min-peak-saved-mib", type=float)
    parser.add_argument("--min-delta-saved-mib", type=float)
    parser.add_argument("--max-peak-ratio", type=float)
    parser.add_argument("--max-delta-ratio", type=float)
    parser.add_argument("--min-pp-ratio", type=float)
    parser.add_argument("--min-tg-ratio", type=float)
    parser.add_argument("--min-measured-over-theory", type=float)
    parser.add_argument("--max-measured-over-theory", type=float)
    parser.add_argument("--fail-empty", action="store_true", help="Fail if no rows match filters.")
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    failures = []
    checked = 0

    for row in rows:
        if not matches(row, set(args.model), set(args.length), set(args.kv)):
            continue
        checked += 1
        label = row.get("label", row.get("kv", "<unknown>"))
        exit_code = as_int(row.get("exit_code"), default=-1)
        if exit_code != 0:
            failures.append(f"{label}: exit_code={exit_code}")
            continue

        check_min(row, "peak_saved_mib_vs_f16", args.min_peak_saved_mib, failures)
        check_min(row, "delta_saved_mib_vs_f16", args.min_delta_saved_mib, failures)
        check_max(row, "peak_ratio_vs_f16", args.max_peak_ratio, failures)
        check_max(row, "delta_ratio_vs_f16", args.max_delta_ratio, failures)
        check_min(row, "pp_ratio_vs_f16", args.min_pp_ratio, failures)
        check_min(row, "tg_ratio_vs_f16", args.min_tg_ratio, failures)
        check_min(row, "measured_saved_over_theory_saved", args.min_measured_over_theory, failures)
        check_max(row, "measured_saved_over_theory_saved", args.max_measured_over_theory, failures)

    if checked == 0 and args.fail_empty:
        failures.append("no rows matched filters")

    if failures:
        print(f"benchmark gate failed: checked={checked}, failures={len(failures)}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"benchmark gate passed: checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
