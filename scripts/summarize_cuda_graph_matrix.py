#!/usr/bin/env python3
"""Summarize cuda_graph_compare_matrix CSVs into README-style numbers (stdout).

Usage:
  python scripts/summarize_cuda_graph_matrix.py results/cuda_graph_compare_matrix/<TAG>
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def _latest_csv(d: Path) -> Path | None:
    if not d.is_dir():
        return None
    cs = sorted(d.glob("bench_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cs[0] if cs else None


def _read_rows(csv_path: Path) -> dict[str, dict[str, float | None]]:
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        out: dict[str, dict[str, float | None]] = {}
        for row in r:
            mode = row["mode"]
            out[mode] = {
                "prefill_median_tok_s": float(row["prefill_median_tok_s"])
                if row.get("prefill_median_tok_s")
                else None,
                "decode_first_tok_s": float(row["decode_first_tok_s"])
                if row.get("decode_first_tok_s")
                else None,
                "decode_steady_median_tok_s": float(row["decode_steady_median_tok_s"])
                if row.get("decode_steady_median_tok_s")
                else None,
                "decode_steady_p95_tok_s": float(row["decode_steady_p95_tok_s"])
                if row.get("decode_steady_p95_tok_s")
                else None,
                "peak_mib_total": float(row["peak_mib_total"])
                if row.get("peak_mib_total")
                else None,
                "kv_k_size_gb": float(row["kv_k_size_gb"]) if row.get("kv_k_size_gb") else None,
                "kv_v_size_gb": float(row["kv_v_size_gb"]) if row.get("kv_v_size_gb") else None,
            }
        return out


def _fmt_num(x: float | None, *, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if nd == 0:
        return str(int(round(x)))
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _mib_kv(d: dict[str, float | None]) -> int | None:
    k, v = d.get("kv_k_size_gb"), d.get("kv_v_size_gb")
    if k is None or v is None:
        return None
    return int(round((k + v) * 1024))


PRESET_TOKENS = {
    "short": 512,
    "medium": 2048,
    "long": 8192,
    "16k": 16384,
    "32k": 32768,
}

# Important metrics first; each title includes higher/lower-better hint for README.
METRIC_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("Decode first (tok/s, higher better)", "decode_first_tok_s", 2, "speed"),
    ("Steady (tok/s, higher better)", "decode_steady_median_tok_s", 2, "speed"),
    ("Peak (MiB, lower better)", "peak_mib_total", 0, "peak"),
    ("KV pool K+V (MiB, measured, lower better)", "", 0, "kv"),
    ("Prefill (tok/s, higher better)", "prefill_median_tok_s", 0, "speed"),
    ("P95 (tok/s, higher better)", "decode_steady_p95_tok_s", 2, "speed"),
)


def _pct_delta(cur: float | None, base: float | None) -> str:
    if cur is None or base is None or base == 0:
        return ""
    p = (cur - base) / base * 100.0
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.2f}%"


def _delta_mib(cur: int | None, base: int | None) -> str:
    if cur is None or base is None:
        return ""
    d = cur - base
    sign = "+" if d > 0 else ""
    return f"{sign}{d} MiB"


def _memory_ratio(cur: int | None, base: int | None) -> str:
    if cur is None or base is None or cur <= 0 or base <= 0:
        return ""
    ratio = max(cur, base) / min(cur, base)
    return f"~{ratio:.2f}x"


def _mode_vs_bf16_pct(
    rows: dict[str, dict[str, float | None]], mode: str, field: str
) -> str:
    bf16 = rows.get("bf16", {}).get(field)
    cur = rows.get(mode, {}).get(field)
    return _pct_delta(cur, bf16)


def _mode_vs_bf16_memory(
    rows: dict[str, dict[str, float | None]], mode: str, *, peak: bool
) -> str:
    bf16 = rows.get("bf16", {})
    cur_row = rows.get(mode, {})
    if peak:
        b_val = bf16.get("peak_mib_total")
        c_val = cur_row.get("peak_mib_total")
        if b_val is None or c_val is None:
            return ""
        b_int, c_int = int(round(b_val)), int(round(c_val))
    else:
        b_int, c_int = _mib_kv(bf16), _mib_kv(cur_row)
        if b_int is None or c_int is None:
            return ""
    pct = _pct_delta(float(c_int), float(b_int))
    dm = _delta_mib(c_int, b_int)
    ratio = _memory_ratio(c_int, b_int)
    details = ", ".join(x for x in (dm, ratio) if x)
    if pct and details:
        return f"{pct} ({details})"
    return pct or details


_TABLE_HEADER = (
    "| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |"
)
_TABLE_RULE = "|---:|---:|---:|---:|---:|---:|"


def _fmt_length_label(length: int) -> str:
    labels = {
        2048: "2K",
        8192: "8K",
        16384: "16K",
        32768: "32K",
    }
    return labels.get(length, str(length))


def _print_table_row(
    length: int,
    *,
    bf16: str,
    oscar: str,
    oscar_delta: str,
    plain: str,
    plain_delta: str,
) -> None:
    length_label = _fmt_length_label(length)
    print(f"| **{length_label}** | {bf16} | {oscar} | {oscar_delta} | {plain} | {plain_delta} |")


def _metric_table(
    title: str,
    field: str,
    *,
    nd: int,
    by_preset: dict[str, dict[str, dict[str, float | None]]],
    presets: list[str],
) -> None:
    print(f"#### {title}\n")
    print(_TABLE_HEADER)
    print(_TABLE_RULE)
    for preset in presets:
        rows = by_preset.get(preset, {})
        length = PRESET_TOKENS[preset]
        bf16 = _fmt_num(rows.get("bf16", {}).get(field), nd=nd)
        oscar = _fmt_num(rows.get("oscar-int2", {}).get(field), nd=nd)
        plain = _fmt_num(rows.get("int2", {}).get(field), nd=nd)
        _print_table_row(
            length,
            bf16=bf16,
            oscar=oscar,
            oscar_delta=_mode_vs_bf16_pct(rows, "oscar-int2", field),
            plain=plain,
            plain_delta=_mode_vs_bf16_pct(rows, "int2", field),
        )
    print()


def _peak_table(by_preset: dict[str, dict[str, dict[str, float | None]]], presets: list[str]) -> None:
    title = next(t for t, _, _, k in METRIC_SPECS if k == "peak")
    print(f"#### {title}\n")
    print(_TABLE_HEADER)
    print(_TABLE_RULE)
    for preset in presets:
        rows = by_preset.get(preset, {})
        length = PRESET_TOKENS[preset]
        bf16_peak = rows.get("bf16", {}).get("peak_mib_total")
        oscar_peak = rows.get("oscar-int2", {}).get("peak_mib_total")
        plain_peak = rows.get("int2", {}).get("peak_mib_total")
        bf16 = str(int(round(bf16_peak))) if bf16_peak is not None else ""
        oscar = str(int(round(oscar_peak))) if oscar_peak is not None else ""
        plain = str(int(round(plain_peak))) if plain_peak is not None else ""
        _print_table_row(
            length,
            bf16=bf16,
            oscar=oscar,
            oscar_delta=_mode_vs_bf16_memory(rows, "oscar-int2", peak=True),
            plain=plain,
            plain_delta=_mode_vs_bf16_memory(rows, "int2", peak=True),
        )
    print()


def _kv_table(by_preset: dict[str, dict[str, dict[str, float | None]]], presets: list[str]) -> None:
    title = next(t for t, _, _, k in METRIC_SPECS if k == "kv")
    print(f"#### {title}\n")
    print(_TABLE_HEADER)
    print(_TABLE_RULE)
    for preset in presets:
        rows = by_preset.get(preset, {})
        length = PRESET_TOKENS[preset]
        bf16 = _mib_kv(rows.get("bf16", {}))
        oscar = _mib_kv(rows.get("oscar-int2", {}))
        plain = _mib_kv(rows.get("int2", {}))
        _print_table_row(
            length,
            bf16=str(bf16) if bf16 is not None else "",
            oscar=str(oscar) if oscar is not None else "",
            oscar_delta=_mode_vs_bf16_memory(rows, "oscar-int2", peak=False),
            plain=str(plain) if plain is not None else "",
            plain_delta=_mode_vs_bf16_memory(rows, "int2", peak=False),
        )
    print()


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base = Path(sys.argv[1])
    presets = ["short", "medium", "long", "16k", "32k"]
    for graph in ("on", "off"):
        by_preset: dict[str, dict[str, dict[str, float | None]]] = {}
        for preset in presets:
            d = base / graph / preset
            csv_path = _latest_csv(d)
            if not csv_path:
                print(f"### {preset}: MISSING\n", file=sys.stderr)
                continue
            by_preset[preset] = _read_rows(csv_path)

        print(f"\n### CUDA graph {graph}\n")
        print(
            "Preset mapping: `512` = `short`, `2K` = `medium`, `8K` = `long`, "
            "`16K` = `16k`, `32K` = `32k`. "
            "Δ vs BF16 is `(mode − BF16) / BF16`; memory rows also show MiB delta.\n"
        )
        for title, field, nd, kind in METRIC_SPECS:
            if kind == "speed":
                _metric_table(title, field, nd=nd, by_preset=by_preset, presets=presets)
            elif kind == "peak":
                _peak_table(by_preset, presets)
            elif kind == "kv":
                _kv_table(by_preset, presets)


if __name__ == "__main__":
    main()
