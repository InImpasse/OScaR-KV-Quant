"""Grid-search oscar-int2 runtime knobs without editing SGLang kernels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oscar_kv_quant.paths import to_repo_relative
from oscar_kv_quant.profiles import PROFILES


@dataclass
class SweepSpec:
    prefix_tokens: int
    recent_tokens: int
    hp_prefix_pool_tokens: int
    hp_max_splits: int
    scale_dtype: str
    fused_rotate: bool
    enable_cuda_graph: bool
    enable_piecewise_cuda_graph: bool


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        out.append(item in {"1", "true", "yes", "on"})
    return out


def iter_specs(args: argparse.Namespace) -> list[SweepSpec]:
    prefixes = _parse_int_list(args.prefix_tokens)
    recents = _parse_int_list(args.recent_tokens)
    hp_prefix_pools = _parse_int_list(args.hp_prefix_pool_tokens)
    splits = _parse_int_list(args.hp_max_splits)
    scales = _parse_str_list(args.scale_dtypes)
    fused = _parse_bool_list(args.fused_rotate)
    graphs = _parse_bool_list(args.cuda_graph)
    piecewise = _parse_bool_list(args.piecewise_cuda_graph)
    specs: list[SweepSpec] = []
    for prefix, recent, hp_pool, split, scale, fr, cg, pwg in itertools.product(
        prefixes, recents, hp_prefix_pools, splits, scales, fused, graphs, piecewise
    ):
        specs.append(
            SweepSpec(
                prefix_tokens=prefix,
                recent_tokens=recent,
                hp_prefix_pool_tokens=hp_pool,
                hp_max_splits=split,
                scale_dtype=scale,
                fused_rotate=fr,
                enable_cuda_graph=cg,
                enable_piecewise_cuda_graph=pwg,
            )
        )
    return specs


def _latest_bench_csv(out_dir: Path) -> Path | None:
    matches = sorted(out_dir.glob("bench_*.csv"))
    return matches[-1] if matches else None


def _read_oscar_row(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") == "oscar-int2" and row.get("server_ok", "").lower() in {
                "true",
                "1",
            }:
                return row
    return None


def _run_one(
    bench_cmd: list[str],
    spec: SweepSpec,
    results_dir: Path,
    dry_run: bool,
) -> tuple[int, str, dict[str, str] | None]:
    tag = (
        f"p{spec.prefix_tokens}_r{spec.recent_tokens}_hp{spec.hp_prefix_pool_tokens}_"
        f"s{spec.hp_max_splits}_{spec.scale_dtype.replace('.', '')}_"
        f"f{int(spec.fused_rotate)}_cg{int(spec.enable_cuda_graph)}_"
        f"pwg{int(spec.enable_piecewise_cuda_graph)}"
    )
    out_dir = results_dir / tag
    cmd = bench_cmd + [
        "--modes",
        "oscar-int2",
        "--prefix-bf16-tokens",
        str(spec.prefix_tokens),
        "--recent-bf16-tokens",
        str(spec.recent_tokens),
        "--hp-prefix-pool-tokens",
        str(spec.hp_prefix_pool_tokens),
        "--mixed-kv-hp-max-splits",
        str(spec.hp_max_splits),
        "--mixed-kv-scale-dtype",
        spec.scale_dtype,
        "--results-dir",
        str(out_dir),
    ]
    if spec.fused_rotate:
        cmd.append("--enable-fused-rotate-clip-quant")
    if spec.enable_cuda_graph:
        cmd.append("--enable-cuda-graph")
    if spec.enable_piecewise_cuda_graph:
        cmd.append("--enable-piecewise-cuda-graph")
    if dry_run:
        print("[sweep:dry-run]", " ".join(cmd), flush=True)
        return 0, tag, None

    proc = subprocess.run(cmd, check=False)
    bench_csv = _latest_bench_csv(out_dir)
    oscar_row = _read_oscar_row(bench_csv) if bench_csv else None
    return proc.returncode, tag, oscar_row


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep oscar-int2 runtime configuration knobs.")
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="granite")
    ap.add_argument("--rot-dir", type=Path, required=True)
    ap.add_argument("--prefill-tokens", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prefix-tokens", default="64")
    ap.add_argument("--recent-tokens", default="128,192,256")
    ap.add_argument("--hp-prefix-pool-tokens", default="64,128")
    ap.add_argument("--hp-max-splits", default="2,4,8")
    ap.add_argument("--scale-dtypes", default="bfloat16")
    ap.add_argument("--fused-rotate", default="true")
    ap.add_argument("--cuda-graph", default="true")
    ap.add_argument("--piecewise-cuda-graph", default="false,true")
    ap.add_argument("--oscar-min-prefill-tokens", type=int, default=0)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch oscar-kv-bench subprocesses.",
    )
    ap.add_argument(
        "--oscar-short-context-fallback",
        choices=["skip", "int2", "bf16"],
        default="skip",
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results/config_sweep"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("bench_args", nargs=argparse.REMAINDER, help="Extra args forwarded to oscar-kv-bench.")
    args = ap.parse_args()

    py = args.python
    bench_cmd = [
        py,
        "-m",
        "oscar_kv_quant.bench",
        "--profile",
        args.profile,
        "--rot-dir",
        str(args.rot_dir),
        "--prefill-tokens",
        str(args.prefill_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.oscar_min_prefill_tokens > 0:
        bench_cmd.extend(
            [
                "--oscar-min-prefill-tokens",
                str(args.oscar_min_prefill_tokens),
                "--oscar-short-context-fallback",
                args.oscar_short_context_fallback,
            ]
        )
    if args.bench_args:
        if args.bench_args[0] == "--":
            args.bench_args = args.bench_args[1:]
        bench_cmd.extend(args.bench_args)

    specs = iter_specs(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "sweep_summary.csv"
    rows: list[dict[str, object]] = []

    for spec in specs:
        rc, tag, oscar_row = _run_one(bench_cmd, spec, args.results_dir, args.dry_run)
        row: dict[str, object] = {
            "tag": tag,
            "returncode": rc,
            "prefix_tokens": spec.prefix_tokens,
            "recent_tokens": spec.recent_tokens,
            "hp_prefix_pool_tokens": spec.hp_prefix_pool_tokens,
            "hp_max_splits": spec.hp_max_splits,
            "scale_dtype": spec.scale_dtype,
            "fused_rotate": spec.fused_rotate,
            "cuda_graph": spec.enable_cuda_graph,
            "piecewise_cuda_graph": spec.enable_piecewise_cuda_graph,
        }
        if oscar_row:
            for key in (
                "request_toks_per_sec",
                "decode_steady_median_tok_s",
                "decode_flush_median_tok_s",
                "effective_decode_tok_s",
                "kv_theory_selected_gib",
            ):
                row[key] = oscar_row.get(key, "")
        rows.append(row)
        print(f"[sweep] {tag} rc={rc}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    pareto: list[dict[str, object]] = []
    for row in rows:
        kv = row.get("kv_theory_selected_gib")
        req = row.get("request_toks_per_sec")
        try:
            kv_f = float(kv) if kv not in ("", None) else None
            req_f = float(req) if req not in ("", None) else None
        except ValueError:
            continue
        if kv_f is not None and req_f is not None and kv_f <= 0.40:
            pareto.append(row)
    if pareto:
        best = max(
            pareto,
            key=lambda r: float(r.get("request_toks_per_sec") or 0.0),
        )
        print(
            "[sweep] pareto best (kv<=0.40 GiB): "
            f"{best.get('tag')} request={best.get('request_toks_per_sec')} tok/s",
            flush=True,
        )

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "num_specs": len(specs),
        "summary_csv": to_repo_relative(summary_path),
        "pareto_candidates": len(pareto),
    }
    (args.results_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[sweep] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
