#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


FIELDS = [
    "variant",
    "status",
    "prompt",
    "kv",
    "kv_pool_mib",
    "peak_mib",
    "delta_mib",
    "max_peak_mib",
    "limit_triggered",
    "exit_code",
    "pp_tps",
    "tg_tps",
    "peak_saved_vs_bf16_mib",
    "kv_saved_vs_bf16_mib",
    "reason",
    "run_dir",
]


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if row.get("variant") == "plain_int3":
            continue
        row["run_dir"] = str(run_dir)
        row["kv"] = f"{row.get('cache_k', '')}/{row.get('cache_v', '')}"
        out.append(row)
    return out


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine llama.cpp KV benchmark summary.csv files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        rows.extend(read_rows(run_dir))

    bf16_by_prompt = {
        r.get("prompt"): r
        for r in rows
        if r.get("variant") == "baseline_bf16" and r.get("prompt")
    }

    for row in rows:
        bf16 = bf16_by_prompt.get(row.get("prompt"))
        bf16_peak = as_float(bf16, "peak_mib") if bf16 else None
        bf16_kv = as_float(bf16, "kv_pool_mib") if bf16 else None
        peak = as_float(row, "peak_mib")
        kv_pool = as_float(row, "kv_pool_mib")
        row["peak_saved_vs_bf16_mib"] = fmt(None if bf16_peak is None or peak is None else bf16_peak - peak)
        row["kv_saved_vs_bf16_mib"] = fmt(None if bf16_kv is None or kv_pool is None else bf16_kv - kv_pool)

    order = {
        "baseline_bf16": 0,
        "oscar_turbo2_streamk": 1,
        "turbo2_streamk": 2,
        "oscar_int4": 3,
        "plain_int4": 4,
        "oscar_int2": 5,
        "plain_int2": 6,
    }
    rows.sort(key=lambda r: (int(r.get("prompt") or 0), order.get(r.get("variant", ""), 99)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "combined.csv"
    md_path = args.out_dir / "combined.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    lines = [
        "| variant | status | prompt | KV | KV MiB | peak MiB | pp tok/s | tg tok/s | peak saved vs BF16 | KV saved vs BF16 | note |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant', '')} | {row.get('status', '')} | {row.get('prompt', '')} | {row.get('kv', '')} | "
            f"{row.get('kv_pool_mib', '')} | {row.get('peak_mib', '')} | {fmt(as_float(row, 'pp_tps'))} | "
            f"{fmt(as_float(row, 'tg_tps'))} | {row.get('peak_saved_vs_bf16_mib', '')} | "
            f"{row.get('kv_saved_vs_bf16_mib', '')} | {row.get('reason', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
