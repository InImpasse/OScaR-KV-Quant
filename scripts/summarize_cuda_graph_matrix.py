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


def _pct_delta(cur: float | None, base: float | None) -> str:
    if cur is None or base is None or base == 0:
        return ""
    p = (cur - base) / base * 100.0
    rounded = int(round(p))
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded}%"


def _mib_kv(d: dict[str, float | None]) -> int | None:
    k, v = d.get("kv_k_size_gb"), d.get("kv_v_size_gb")
    if k is None or v is None:
        return None
    return int(round((k + v) * 1024))


def _delta_mib(cur: int | None, base: int | None) -> str:
    if cur is None or base is None:
        return ""
    d = cur - base
    sign = "+" if d > 0 else ""
    return f"{sign}{d} MiB"


def _pct_kv(cur: int | None, base: int | None) -> str:
    if cur is None or base is None or base == 0:
        return ""
    p = (cur - base) / base * 100.0
    rounded = int(round(p))
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded}%"


def _peak_lines(m: str, rows: dict[str, dict[str, float | None]], bf16_peak: float | None) -> str:
    d = rows.get(m, {})
    peak = d.get("peak_mib_total")
    if peak is None:
        return ""
    p_int = int(round(peak))
    if m == "bf16":
        return str(p_int)
    if bf16_peak is None or bf16_peak == 0:
        return str(p_int)
    dp = (peak - bf16_peak) / bf16_peak * 100.0
    dm = int(round(peak - bf16_peak))
    ps = int(round(dp))
    signp = "+" if ps > 0 else ""
    signm = "+" if dm > 0 else ""
    return f"{p_int}<br>({signp}{ps}%)<br>({signm}{dm} MiB)"


def _kv_lines(m: str, rows: dict[str, dict[str, float | None]], bf16_kv: int | None) -> str:
    mib = _mib_kv(rows.get(m, {}))
    if mib is None:
        return ""
    if m == "bf16":
        return f"{mib} MiB"
    if bf16_kv is None or bf16_kv == 0:
        return f"{mib} MiB"
    dp = _pct_kv(mib, bf16_kv)
    dm = _delta_mib(mib, bf16_kv)
    return f"{mib} MiB<br>({dp})<br>({dm})"


def _throughput_cell(
    m: str,
    field: str,
    rows: dict[str, dict[str, float | None]],
    bf16: float | None,
    *,
    nd: int,
) -> str:
    cur = rows.get(m, {}).get(field)
    if cur is None:
        return ""
    cell = _fmt_num(cur, nd=nd)
    if m == "bf16":
        return cell
    if bf16 is None or bf16 == 0:
        return cell
    pc = _pct_delta(cur, bf16)
    return f"{cell}<br>({pc})"


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base = Path(sys.argv[1])
    presets = ["short", "medium", "long", "16k", "32k"]
    for graph in ("on", "off"):
        print(f"\n## CUDA graph {graph}\n")
        for preset in presets:
            d = base / graph / preset
            csv_path = _latest_csv(d)
            if not csv_path:
                print(f"### {preset}: MISSING\n")
                continue
            rows = _read_rows(csv_path)
            b = rows.get("bf16", {})
            bf16_peak = b.get("peak_mib_total")
            bf16_kv = _mib_kv(b)
            print(f"### {preset} ({csv_path.name})\n")
            print("| Mode | Prefill | Decode1 | Steady | P95 | Peak | KV MiB |")
            print("|------|---------|---------|--------|-----|------|--------|")
            for label, m in (
                ("BF16", "bf16"),
                ("INT2", "int2"),
                ("OSCAR", "oscar-int2"),
            ):
                parts = [
                    _throughput_cell(
                        m,
                        "prefill_median_tok_s",
                        rows,
                        b.get("prefill_median_tok_s"),
                        nd=0,
                    ),
                    _throughput_cell(
                        m, "decode_first_tok_s", rows, b.get("decode_first_tok_s"), nd=2
                    ),
                    _throughput_cell(
                        m,
                        "decode_steady_median_tok_s",
                        rows,
                        b.get("decode_steady_median_tok_s"),
                        nd=2,
                    ),
                    _throughput_cell(
                        m,
                        "decode_steady_p95_tok_s",
                        rows,
                        b.get("decode_steady_p95_tok_s"),
                        nd=2,
                    ),
                ]
                peak_s = _peak_lines(m, rows, bf16_peak)
                kv_s = _kv_lines(m, rows, bf16_kv)
                print(f"| {label} | " + " | ".join(parts) + f" | {peak_s} | {kv_s} |")
            print()


if __name__ == "__main__":
    main()
