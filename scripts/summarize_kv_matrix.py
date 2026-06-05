#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


KV_ATOM = r"(?:f16|q8_0|q4_0|q2_0|q2|q2hp|q2_0_hp|q2_0_owht|q2_0_owht_nohad|q2_0_owht_clip|q2_0_owht_nohad_clip)"
KV_PATTERN = rf"(?:{KV_ATOM}|k{KV_ATOM}_v{KV_ATOM})"
LABEL_RE = re.compile(rf"^(?P<model>.+?)_(?P<kv>{KV_PATTERN})_(?P<length>short|medium|long)_p(?P<prompt>\d+)_n(?P<gen>\d+)$")
ALT_LABEL_RE = re.compile(rf"^(?P<model>.+?)_(?P<length>short|medium|long)_(?P<kv>{KV_PATTERN})_p(?P<prompt>\d+)_n(?P<gen>\d+)$")


def parse_summary(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_bench_json(run_dir: Path, label: str, summary: dict[str, str]):
    candidates = [
        Path(summary["stdout"]) if "stdout" in summary else None,
        run_dir / f"{label}.stdout.txt",
        run_dir / f"{label}.json",
    ]
    for path in candidates:
        if path and path.exists():
            text = path.read_text().strip()
            if text.startswith("["):
                return json.loads(text)
    return []


def normalize_kv(kv: str) -> str:
    if kv == "q2":
        return "q2_0"
    if kv == "q2hp":
        return "q2_0_hp"
    return kv


def storage_kv(kv: str) -> str:
    if kv.startswith("q2_0_owht"):
        return "q2_0"
    if kv.startswith("k") and "_v" in kv:
        k_part, v_part = kv[1:].split("_v", 1)
        return f"k{storage_kv(k_part)}_v{storage_kv(v_part)}"
    return kv


def split_kv(kv: str) -> tuple[str, str, str]:
    if kv.startswith("k") and "_v" in kv:
        k_part, v_part = kv[1:].split("_v", 1)
        kv_k = normalize_kv(k_part)
        kv_v = normalize_kv(v_part)
        return f"k{kv_k}_v{kv_v}", kv_k, kv_v
    kv_norm = normalize_kv(kv)
    return kv_norm, kv_norm, kv_norm


def parse_label(label: str):
    match = LABEL_RE.match(label) or ALT_LABEL_RE.match(label)
    if not match:
        return None
    row = match.groupdict()
    row["kv"], row["kv_k"], row["kv_v"] = split_kv(row["kv"])
    row["prompt"] = int(row["prompt"])
    row["gen"] = int(row["gen"])
    return row


def bench_rates(entries):
    pp = None
    tg = None
    for item in entries:
        n_prompt = int(item.get("n_prompt", 0))
        n_gen = int(item.get("n_gen", 0))
        avg_ts = float(item.get("avg_ts", 0.0))
        if n_prompt > 0 and n_gen == 0:
            pp = avg_ts
        if n_gen > 0:
            tg = avg_ts
    return pp, tg


def pct(value):
    if value is None:
        return ""
    return f"{value:.1f}"


def pct_ratio(value):
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def ratio(value, base):
    if value is None or base in (None, 0):
        return None
    return value / base


def load_theory_csv(path: Path):
    theory = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kv_type = normalize_kv(row["kv_type"])
            key = (row["model"], int(row["context"]), kv_type)
            theory[key] = {
                "theory_kv_mib": float(row["kv_mib"]),
                "theory_saved_mib_vs_f16": float(row["saved_mib_vs_f16"]),
                "theory_ratio_vs_f16": float(row["ratio_vs_f16"]),
            }
            if "hp_slots" in row:
                theory[key]["theory_hp_slots"] = int(row["hp_slots"])
    return theory


def add_theory(rows, theory):
    if not theory:
        return rows

    for row in rows:
        values = theory.get((row["model"], row["prompt"], row["kv"]))
        if not values:
            values = theory.get((row["model"], row["prompt"], storage_kv(row["kv"])))
        if not values:
            continue
        row.update(values)
        theory_saved = row.get("theory_saved_mib_vs_f16")
        if theory_saved is not None and theory_saved > 0:
            row["measured_saved_over_theory_saved"] = ratio(
                row.get("peak_saved_mib_vs_f16"),
                theory_saved,
            )
    return rows


def collect(run_dir: Path):
    rows = []
    for summary_path in sorted(run_dir.glob("*.summary.txt")):
        summary = parse_summary(summary_path)
        label = summary.get("label", summary_path.name.removesuffix(".summary.txt"))
        parsed = parse_label(label)
        if parsed is None:
            continue
        entries = load_bench_json(run_dir, label, summary)
        pp_tps, tg_tps = bench_rates(entries)
        rows.append({
            **parsed,
            "label": label,
            "exit_code": int(summary.get("exit_code", -1)),
            "baseline_mib": int(summary.get("baseline_mib", 0)),
            "peak_mib": int(summary.get("peak_mib", 0)),
            "delta_mib": int(summary.get("delta_mib", 0)),
            "duration_ms": int(summary.get("duration_ms", 0)),
            "cache_type_k": summary.get("cache_type_k", ""),
            "cache_type_v": summary.get("cache_type_v", ""),
            "llama_kv_hp_sink": summary.get("llama_kv_hp_sink", ""),
            "llama_kv_hp_recent": summary.get("llama_kv_hp_recent", ""),
            "llama_kv_q2_0_owht": summary.get("llama_kv_q2_0_owht", ""),
            "llama_kv_no_hadamard": summary.get("llama_kv_no_hadamard", ""),
            "llama_kv_clip_ratio": summary.get("llama_kv_clip_ratio", ""),
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
        })
    return rows


def add_baselines(rows):
    baselines = {}
    for row in rows:
        if row["kv"] == "f16":
            baselines[(row["model"], row["length"])] = row

    for row in rows:
        base = baselines.get((row["model"], row["length"]))
        row["peak_ratio_vs_f16"] = ratio(row["peak_mib"], base["peak_mib"] if base else None)
        row["peak_saved_mib_vs_f16"] = (base["peak_mib"] - row["peak_mib"]) if base else None
        row["delta_ratio_vs_f16"] = ratio(row["delta_mib"], base["delta_mib"] if base else None)
        row["delta_saved_mib_vs_f16"] = (base["delta_mib"] - row["delta_mib"]) if base else None
        row["pp_ratio_vs_f16"] = ratio(row["pp_tps"], base["pp_tps"] if base else None)
        row["tg_ratio_vs_f16"] = ratio(row["tg_tps"], base["tg_tps"] if base else None)
    return rows


def write_csv(rows, path: Path):
    fields = [
        "model", "length", "prompt", "gen", "kv", "kv_k", "kv_v",
        "cache_type_k", "cache_type_v", "llama_kv_hp_sink", "llama_kv_hp_recent",
        "llama_kv_q2_0_owht", "llama_kv_no_hadamard", "llama_kv_clip_ratio",
        "baseline_mib", "peak_mib",
        "peak_saved_mib_vs_f16", "peak_ratio_vs_f16", "delta_mib",
        "delta_saved_mib_vs_f16", "delta_ratio_vs_f16",
        "theory_kv_mib", "theory_saved_mib_vs_f16",
        "theory_ratio_vs_f16", "measured_saved_over_theory_saved",
        "pp_tps", "pp_ratio_vs_f16", "tg_tps", "tg_ratio_vs_f16",
        "duration_ms", "exit_code", "label",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown(rows, path: Path):
    lines = [
        "| model | length | KV | K cache | V cache | baseline MiB | peak MiB | peak saved | delta MiB | delta saved | delta ratio | theory KV MiB | theory saved | meas/theory saved | pp tok/s | pp ratio | tg tok/s | tg ratio |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['length']} | {row['kv']} | "
            f"{row['kv_k']} | {row['kv_v']} | "
            f"{row['baseline_mib']} | {row['peak_mib']} | "
            f"{row['peak_saved_mib_vs_f16'] if row['peak_saved_mib_vs_f16'] is not None else ''} | "
            f"{row['delta_mib']} | {row['delta_saved_mib_vs_f16'] if row['delta_saved_mib_vs_f16'] is not None else ''} | "
            f"{pct_ratio(row['delta_ratio_vs_f16'])} | "
            f"{pct(row.get('theory_kv_mib'))} | {pct(row.get('theory_saved_mib_vs_f16'))} | "
            f"{pct_ratio(row.get('measured_saved_over_theory_saved'))} | "
            f"{pct(row['pp_tps'])} | {pct_ratio(row['pp_ratio_vs_f16'])} | "
            f"{pct(row['tg_tps'])} | {pct_ratio(row['tg_ratio_vs_f16'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize llama-bench KV matrix runs with VRAM summaries.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--theory-csv", type=Path, help="Optional estimate_kv_cache.py CSV to join by model/context/KV type.")
    args = parser.parse_args()

    run_dir = args.run_dir
    rows = add_baselines(collect(run_dir))
    rows = add_theory(rows, load_theory_csv(args.theory_csv) if args.theory_csv else None)
    rows.sort(key=lambda r: (r["model"], r["prompt"], r["kv"]))

    out_prefix = args.out_prefix or run_dir / "kv_matrix_summary"
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"))
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
