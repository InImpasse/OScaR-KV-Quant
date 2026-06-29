#!/usr/bin/env python3
"""Summarize llama.cpp KV preset matrices in the SGLang harness table shape."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PRESET_TOKENS = {
    "short": 512,
    "medium": 2048,
    "long": 8192,
    "16k": 16384,
    "32k": 32768,
}
TOKEN_PRESET = {tokens: preset for preset, tokens in PRESET_TOKENS.items()}

VARIANT_MODE = {
    "baseline_bf16": "bf16",
    "oscar_int2": "oscar-int2",
    "plain_int2": "int2",
    "oscar_int4": "oscar-int4",
    "plain_int4": "int4",
}

MODE_LABEL = {
    "bf16": "BF16",
    "oscar-int2": "OSCAR INT2",
    "int2": "plain INT2",
    "oscar-int4": "OSCAR INT4",
    "int4": "plain INT4",
}


def _read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        mode = VARIANT_MODE.get(row.get("variant", ""))
        if mode:
            out[mode] = row
    return out


def _matrix(base: Path) -> dict[str, dict[str, dict[str, str]]]:
    return {preset: _read_summary(base / preset / "summary.csv") for preset in PRESET_TOKENS}


def _preset_from_row(row: dict[str, str], source: Path) -> str:
    preset = row.get("preset", "").strip()
    if preset:
        if preset not in PRESET_TOKENS:
            raise SystemExit(f"{source}: unsupported preset {preset!r}")
        return preset
    raw_tokens = (row.get("prefill_tokens") or row.get("prompt") or row.get("length_tokens") or "").strip()
    if not raw_tokens:
        raise SystemExit(f"{source}: manual metrics row needs preset or prefill_tokens")
    try:
        tokens = int(raw_tokens)
    except ValueError as exc:
        raise SystemExit(f"{source}: invalid prefill_tokens {raw_tokens!r}") from exc
    if tokens not in TOKEN_PRESET:
        raise SystemExit(f"{source}: unsupported prefill_tokens {tokens}; expected one of {sorted(TOKEN_PRESET)}")
    return TOKEN_PRESET[tokens]


def _mode_from_row(row: dict[str, str], source: Path) -> str:
    mode = (row.get("mode") or row.get("variant") or "").strip()
    aliases = {
        "baseline_bf16": "bf16",
        "bf16": "bf16",
        "oscar_int2": "oscar-int2",
        "oscar-int2": "oscar-int2",
        "plain_int2": "int2",
        "int2": "int2",
        "oscar_int4": "oscar-int4",
        "oscar-int4": "oscar-int4",
        "plain_int4": "int4",
        "int4": "int4",
    }
    if mode not in aliases:
        raise SystemExit(f"{source}: unsupported mode/variant {mode!r}")
    return aliases[mode]


def _apply_manual_metrics(matrix: dict[str, dict[str, dict[str, str]]], csv_path: Path | None) -> None:
    if csv_path is None:
        return
    if not csv_path.exists():
        raise SystemExit(f"manual metrics CSV not found: {csv_path}")
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        preset = _preset_from_row(row, csv_path)
        mode = _mode_from_row(row, csv_path)
        target = matrix.setdefault(preset, {}).setdefault(mode, {"status": "manual", "variant": mode})
        for field in ("decode_first_tok_s", "decode_steady_p95_tok_s"):
            value = row.get(field, "").strip()
            if value:
                target[field] = value
        target["manual_metrics_source"] = str(csv_path)


def _f(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    raw = row.get(key, "")
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fmt_num(value: float | None, nd: int) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return ""
    if nd == 0:
        return str(int(round(value)))
    return f"{value:.{nd}f}"


def _pct(cur: float | None, base: float | None) -> str:
    if cur is None or base is None or base == 0:
        return ""
    value = (cur - base) / base * 100.0
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.0f}%"


def _mem_delta(cur: float | None, base: float | None) -> str:
    if cur is None or base is None:
        return ""
    pct = _pct(cur, base)
    delta = int(round(cur - base))
    sign = "+" if delta > 0 else ""
    return f"{pct} ({sign}{delta} MiB)" if pct else f"{sign}{delta} MiB"


def _length(preset: str) -> str:
    n = PRESET_TOKENS[preset]
    return {2048: "2K", 8192: "8K", 16384: "16K", 32768: "32K"}.get(n, str(n))


def _table(
    title: str,
    matrix: dict[str, dict[str, dict[str, str]]],
    *,
    field: str,
    nd: int,
    memory: bool = False,
) -> list[str]:
    lines = [
        title,
        "Length (tokens)\tBF16\tOSCAR INT2\tDelta vs BF16\tplain INT2\tDelta vs BF16",
    ]
    for preset in PRESET_TOKENS:
        rows = matrix.get(preset, {})
        bf16 = _f(rows.get("bf16"), field)
        oscar = _f(rows.get("oscar-int2"), field)
        plain = _f(rows.get("int2"), field)
        delta_fn = _mem_delta if memory else _pct
        lines.append(
            "\t".join(
                [
                    _length(preset),
                    _fmt_num(bf16, nd),
                    _fmt_num(oscar, nd),
                    delta_fn(oscar, bf16),
                    _fmt_num(plain, nd),
                    delta_fn(plain, bf16),
                ]
            )
        )
    lines.append("")
    return lines


def _markdown_table(
    title: str,
    matrix: dict[str, dict[str, dict[str, str]]],
    *,
    field: str,
    nd: int,
    memory: bool = False,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for preset in PRESET_TOKENS:
        rows = matrix.get(preset, {})
        bf16 = _f(rows.get("bf16"), field)
        oscar = _f(rows.get("oscar-int2"), field)
        plain = _f(rows.get("int2"), field)
        delta_fn = _mem_delta if memory else _pct
        lines.append(
            f"| {_length(preset)} | {_fmt_num(bf16, nd)} | {_fmt_num(oscar, nd)} | "
            f"{delta_fn(oscar, bf16)} | {_fmt_num(plain, nd)} | {delta_fn(plain, bf16)} |"
        )
    lines.append("")
    return lines


def _write_compatible_csv(base: Path, matrix: dict[str, dict[str, dict[str, str]]]) -> None:
    fields = [
        "preset",
        "mode",
        "variant",
        "status",
        "prefill_tokens",
        "max_new_tokens",
        "prefill_median_tok_s",
        "decode_first_tok_s",
        "decode_steady_median_tok_s",
        "decode_steady_p95_tok_s",
        "peak_mib_total",
        "kv_pool_mib",
        "kv_k_size_gb",
        "kv_v_size_gb",
        "server_ok",
        "error",
        "manual_metrics_source",
        "run_dir",
    ]
    with (base / "bench_compatible.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for preset, rows in matrix.items():
            for mode, row in rows.items():
                kv_pool = _f(row, "kv_pool_mib")
                kv_half_gb = None if kv_pool is None else kv_pool / 2.0 / 1024.0
                writer.writerow(
                    {
                        "preset": preset,
                        "mode": mode,
                        "variant": row.get("variant", ""),
                        "status": row.get("status", ""),
                        "prefill_tokens": row.get("prompt", ""),
                        "max_new_tokens": row.get("gen", ""),
                        "prefill_median_tok_s": row.get("pp_tps", ""),
                        "decode_first_tok_s": row.get("decode_first_tok_s", ""),
                        "decode_steady_median_tok_s": row.get("tg_tps", ""),
                        "decode_steady_p95_tok_s": row.get("decode_steady_p95_tok_s", ""),
                        "peak_mib_total": row.get("peak_mib", ""),
                        "kv_pool_mib": row.get("kv_pool_mib", ""),
                        "kv_k_size_gb": "" if kv_half_gb is None else f"{kv_half_gb:.6f}",
                        "kv_v_size_gb": "" if kv_half_gb is None else f"{kv_half_gb:.6f}",
                        "server_ok": "true" if row.get("status") == "ok" else "false",
                        "error": row.get("reason", ""),
                        "manual_metrics_source": row.get("manual_metrics_source", ""),
                        "run_dir": str(base / preset),
                    }
                )


def _summary_lines(matrix: dict[str, dict[str, dict[str, str]]], markdown: bool) -> list[str]:
    make = _markdown_table if markdown else _table
    lines: list[str] = []
    lines += make("Decode first (tok/s, higher better)", matrix, field="decode_first_tok_s", nd=2)
    lines += make("Steady (tok/s, higher better)", matrix, field="tg_tps", nd=2)
    lines += make("Peak (MiB, lower better)", matrix, field="peak_mib", nd=0, memory=True)
    lines += make("KV pool K+V (MiB, measured/estimated, lower better)", matrix, field="kv_pool_mib", nd=0, memory=True)
    lines += make("Prefill (tok/s, higher better)", matrix, field="pp_tps", nd=0)
    lines += make("P95 (tok/s, higher better)", matrix, field="decode_steady_p95_tok_s", nd=2)
    note = (
        "Note: Steady uses llama-bench tg tok/s, and Prefill uses pp tok/s. "
        "Decode first/P95 are populated only when provided through manual_metrics.csv "
        "or --manual-metrics."
    )
    if markdown:
        lines += ["### Notes", "", note, ""]
    else:
        lines += [note, ""]
    return lines


def _write_single(base: Path, name: str = "matrix", manual_metrics: Path | None = None) -> None:
    matrix = _matrix(base)
    if manual_metrics is None and (base / "manual_metrics.csv").exists():
        manual_metrics = base / "manual_metrics.csv"
    _apply_manual_metrics(matrix, manual_metrics)
    (base / f"{name}.txt").write_text("\n".join(_summary_lines(matrix, markdown=False)) + "\n")
    (base / f"{name}.md").write_text("\n".join(_summary_lines(matrix, markdown=True)) + "\n")
    _write_compatible_csv(base, matrix)
    print(f"wrote {base / (name + '.txt')}")
    print(f"wrote {base / (name + '.md')}")
    print(f"wrote {base / 'bench_compatible.csv'}")


def _write_graph_compare(base: Path, manual_metrics: Path | None = None) -> None:
    for graph in ("on", "off"):
        graph_dir = base / graph
        if graph_dir.is_dir():
            graph_manual = manual_metrics
            if graph_manual is None and (graph_dir / "manual_metrics.csv").exists():
                graph_manual = graph_dir / "manual_metrics.csv"
            _write_single(graph_dir, name=f"matrix_{graph}", manual_metrics=graph_manual)

    lines: list[str] = ["Derived: toggling CUDA graph (on vs off)", ""]
    on = _matrix(base / "on")
    off = _matrix(base / "off")
    _apply_manual_metrics(on, manual_metrics if manual_metrics is not None else None)
    _apply_manual_metrics(off, manual_metrics if manual_metrics is not None else None)
    lines.append("Prefill\tBF16 steady x\tOSCAR steady x\tplain INT2 steady x\tBF16 peak Delta\tOSCAR peak Delta\tplain INT2 peak Delta")
    for preset in PRESET_TOKENS:
        row = [_length(preset)]
        for mode in ("bf16", "oscar-int2", "int2"):
            on_tg = _f(on.get(preset, {}).get(mode), "tg_tps")
            off_tg = _f(off.get(preset, {}).get(mode), "tg_tps")
            row.append("" if on_tg is None or off_tg in (None, 0) else f"{on_tg / off_tg:.2f}x")
        for mode in ("bf16", "oscar-int2", "int2"):
            on_peak = _f(on.get(preset, {}).get(mode), "peak_mib")
            off_peak = _f(off.get(preset, {}).get(mode), "peak_mib")
            row.append("" if on_peak is None or off_peak is None else f"{int(round(on_peak - off_peak)):+d} MiB")
        lines.append("\t".join(row))
    lines.append("")
    (base / "graph_toggle_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {base / 'graph_toggle_summary.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--graph-compare", action="store_true")
    parser.add_argument(
        "--manual-metrics",
        type=Path,
        help=(
            "Optional CSV overlay with columns preset or prefill_tokens, mode or variant, "
            "decode_first_tok_s, decode_steady_p95_tok_s."
        ),
    )
    args = parser.parse_args()

    if args.graph_compare:
        _write_graph_compare(args.matrix_dir, manual_metrics=args.manual_metrics)
    else:
        _write_single(args.matrix_dir, manual_metrics=args.manual_metrics)


if __name__ == "__main__":
    main()
