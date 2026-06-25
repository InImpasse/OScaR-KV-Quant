#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


FIELDS = [
    "mode",
    "opt",
    "variant",
    "status",
    "prompt",
    "kv",
    "peak_mib",
    "pp_tps",
    "tg_tps",
    "pp_delta_vs_off",
    "pp_pct_vs_off",
    "tg_delta_vs_off",
    "tg_pct_vs_off",
    "run_dir",
]


def read_case(run_dir: Path) -> dict[str, str]:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("variant") != "plain_int3"]
    if len(rows) != 1:
        raise ValueError(f"expected one measured row in {path}, got {len(rows)}")
    return rows[0]


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def pct(value: float | None, base: float | None) -> str:
    if value is None or base is None or base == 0:
        return ""
    return f"{100.0 * value / base:.2f}"


def build_row(root: Path, mode: str, opt: str) -> dict[str, str]:
    run_dir = root / f"{mode}_opt{opt}"
    row = read_case(run_dir)
    return {
        "mode": mode,
        "opt": opt,
        "variant": row.get("variant", ""),
        "status": row.get("status", ""),
        "prompt": row.get("prompt", ""),
        "kv": f"{row.get('cache_k', '')}/{row.get('cache_v', '')}",
        "peak_mib": row.get("peak_mib", ""),
        "pp_tps": row.get("pp_tps", ""),
        "tg_tps": row.get("tg_tps", ""),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a CUDA graph off/on A/B run.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    rows = [
        build_row(args.run_root, "off", "0"),
        build_row(args.run_root, "on", "1"),
    ]

    baseline = rows[0]
    base_pp = as_float(baseline, "pp_tps")
    base_tg = as_float(baseline, "tg_tps")
    for row in rows:
        pp = as_float(row, "pp_tps")
        tg = as_float(row, "tg_tps")
        pp_delta = None if pp is None or base_pp is None else pp - base_pp
        tg_delta = None if tg is None or base_tg is None else tg - base_tg
        row["pp_delta_vs_off"] = fmt(pp_delta)
        row["pp_pct_vs_off"] = pct(pp_delta, base_pp)
        row["tg_delta_vs_off"] = fmt(tg_delta)
        row["tg_pct_vs_off"] = pct(tg_delta, base_tg)

    csv_path = args.run_root / "graph_ab.csv"
    md_path = args.run_root / "graph_ab.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    lines = [
        "| mode | opt | variant | prompt | KV | peak MiB | pp tok/s | pp delta | pp delta % | tg tok/s | tg delta | tg delta % |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('mode', '')} | {row.get('opt', '')} | {row.get('variant', '')} | "
            f"{row.get('prompt', '')} | {row.get('kv', '')} | {row.get('peak_mib', '')} | "
            f"{fmt(as_float(row, 'pp_tps'))} | {row.get('pp_delta_vs_off', '')} | "
            f"{row.get('pp_pct_vs_off', '')} | {fmt(as_float(row, 'tg_tps'))} | "
            f"{row.get('tg_delta_vs_off', '')} | {row.get('tg_pct_vs_off', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
