#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


KV_ATOM = r"(?:f16|q8_0|q4_0|q2_0|q2|q2hp|q2_0_hp|q2_0_owht|q2_0_owht_nohad|q2_0_owht_clip|q2_0_owht_nohad_clip)"
KV_PATTERN = rf"(?:{KV_ATOM}|k{KV_ATOM}_v{KV_ATOM})"
LABEL_RE = re.compile(
    rf"^(?P<model>.+?)_(?P<length>short|medium|long)_(?P<kv>{KV_PATTERN})_c(?P<context>\d+)_chunks(?P<chunks>-?\d+)$"
)
PPL_RE = re.compile(r"(?:Final estimate:\s*)?PPL\s*=\s*(?P<ppl>[0-9]+(?:\.[0-9]+)?)(?:\s*\+/-\s*(?P<unc>[0-9]+(?:\.[0-9]+)?))?")


def normalize_kv(kv: str) -> str:
    if kv == "q2":
        return "q2_0"
    if kv == "q2hp":
        return "q2_0_hp"
    return kv


def split_kv(kv: str) -> tuple[str, str, str]:
    if kv.startswith("k") and "_v" in kv:
        k_part, v_part = kv[1:].split("_v", 1)
        kv_k = normalize_kv(k_part)
        kv_v = normalize_kv(v_part)
        return f"k{kv_k}_v{kv_v}", kv_k, kv_v
    kv_norm = normalize_kv(kv)
    return kv_norm, kv_norm, kv_norm


def parse_summary(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_label(label: str):
    match = LABEL_RE.match(label)
    if not match:
        return None
    row = match.groupdict()
    row["kv"], row["kv_k"], row["kv_v"] = split_kv(row["kv"])
    row["context"] = int(row["context"])
    row["chunks"] = int(row["chunks"])
    return row


def read_text(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="replace")


def extract_ppl(text: str):
    matches = list(PPL_RE.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    ppl = float(match.group("ppl"))
    unc = float(match.group("unc")) if match.group("unc") else None
    return ppl, unc


def ratio(value, base):
    if value is None or base in (None, 0):
        return None
    return value / base


def pct_ratio(value):
    if value is None:
        return ""
    return f"{100.0 * value:.2f}%"


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def collect(run_dir: Path):
    rows = []
    for summary_path in sorted(run_dir.glob("*.summary.txt")):
        summary = parse_summary(summary_path)
        label = summary.get("label", summary_path.name.removesuffix(".summary.txt"))
        parsed = parse_label(label)
        if parsed is None:
            continue

        text = read_text(summary.get("stdout", "")) + "\n" + read_text(summary.get("stderr", ""))
        ppl, ppl_unc = extract_ppl(text)
        rows.append({
            **parsed,
            "label": label,
            "exit_code": int(summary.get("exit_code", -1)),
            "duration_ms": int(summary.get("duration_ms", 0)),
            "cache_type_k": summary.get("cache_type_k", ""),
            "cache_type_v": summary.get("cache_type_v", ""),
            "llama_kv_hp_sink": summary.get("llama_kv_hp_sink", ""),
            "llama_kv_hp_recent": summary.get("llama_kv_hp_recent", ""),
            "llama_kv_q2_0_owht": summary.get("llama_kv_q2_0_owht", ""),
            "llama_kv_no_hadamard": summary.get("llama_kv_no_hadamard", ""),
            "llama_kv_clip_ratio": summary.get("llama_kv_clip_ratio", ""),
            "baseline_mib": int(summary.get("baseline_mib", 0)) if "baseline_mib" in summary else None,
            "peak_mib": int(summary.get("peak_mib", 0)) if "peak_mib" in summary else None,
            "delta_mib": int(summary.get("delta_mib", 0)) if "delta_mib" in summary else None,
            "ppl": ppl,
            "ppl_unc": ppl_unc,
        })
    return rows


def add_baselines(rows):
    baselines = {}
    for row in rows:
        if row["kv"] == "f16":
            baselines[(row["model"], row["length"], row["context"])] = row

    for row in rows:
        base = baselines.get((row["model"], row["length"], row["context"]))
        base_ppl = base["ppl"] if base else None
        row["ppl_ratio_vs_f16"] = ratio(row["ppl"], base_ppl)
        row["ppl_delta_vs_f16"] = (row["ppl"] - base_ppl) if row["ppl"] is not None and base_ppl is not None else None
        row["peak_saved_mib_vs_f16"] = (
            base["peak_mib"] - row["peak_mib"]
            if base and base.get("peak_mib") is not None and row.get("peak_mib") is not None
            else None
        )
        row["delta_saved_mib_vs_f16"] = (
            base["delta_mib"] - row["delta_mib"]
            if base and base.get("delta_mib") is not None and row.get("delta_mib") is not None
            else None
        )
    return rows


def write_csv(rows, path: Path):
    fields = [
        "model", "length", "context", "chunks", "kv", "kv_k", "kv_v",
        "cache_type_k", "cache_type_v", "llama_kv_hp_sink", "llama_kv_hp_recent",
        "llama_kv_q2_0_owht", "llama_kv_no_hadamard", "llama_kv_clip_ratio",
        "ppl", "ppl_unc",
        "ppl_delta_vs_f16", "ppl_ratio_vs_f16", "baseline_mib", "peak_mib",
        "peak_saved_mib_vs_f16", "delta_mib", "delta_saved_mib_vs_f16",
        "duration_ms", "exit_code", "label",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown(rows, path: Path):
    lines = [
        "| model | length | context | KV | K cache | V cache | exit | PPL | delta vs f16 | ratio vs f16 | baseline MiB | peak MiB | peak saved | delta MiB | delta saved |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['length']} | {row['context']} | {row['kv']} | "
            f"{row['kv_k']} | {row['kv_v']} | "
            f"{row.get('exit_code', '')} | "
            f"{fmt(row.get('ppl'))} | {fmt(row.get('ppl_delta_vs_f16'))} | {pct_ratio(row.get('ppl_ratio_vs_f16'))} | "
            f"{row.get('baseline_mib') if row.get('baseline_mib') is not None else ''} | "
            f"{row.get('peak_mib') if row.get('peak_mib') is not None else ''} | "
            f"{row.get('peak_saved_mib_vs_f16') if row.get('peak_saved_mib_vs_f16') is not None else ''} | "
            f"{row.get('delta_mib') if row.get('delta_mib') is not None else ''} | "
            f"{row.get('delta_saved_mib_vs_f16') if row.get('delta_saved_mib_vs_f16') is not None else ''} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize KV-cache perplexity matrix runs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-prefix", type=Path)
    args = parser.parse_args()

    rows = add_baselines(collect(args.run_dir))
    rows.sort(key=lambda r: (r["model"], r["context"], r["kv"]))

    out_prefix = args.out_prefix or args.run_dir / "kv_ppl_summary"
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"))
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
