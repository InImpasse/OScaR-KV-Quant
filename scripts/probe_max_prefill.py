#!/usr/bin/env python3
"""Find max successful --prefill-tokens per KV mode via oscar-kv-bench (binary search)."""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path


def _latest_csv(results_dir: Path) -> Path | None:
    paths = sorted(results_dir.glob("bench_granite_*.csv"))
    return paths[-1] if paths else None


def _ensure_results_parent(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def trial(
    *,
    root: Path,
    bench_sh: Path,
    mode: str,
    prefill: int,
    results_dir: Path,
    rot_dir: Path | None,
    max_new_tokens: int,
    health_timeout: float,
    mem_frac: float,
    triton_int2: bool,
    port: int,
) -> tuple[bool, str]:
    _ensure_results_parent(results_dir)
    before = {_latest_csv(results_dir)}
    cmd: list[str] = [
        str(bench_sh),
        "--profile",
        "granite",
        "--modes",
        mode,
        "--prefill-tokens",
        str(prefill),
        "--max-new-tokens",
        str(max_new_tokens),
        "--warmup-requests",
        "0",
        "--bench-requests",
        "1",
        "--request-api",
        "completions",
        "--results-dir",
        str(results_dir),
        "--health-timeout",
        str(health_timeout),
        "--mem-fraction-static",
        str(mem_frac),
        "--port",
        str(port),
    ]
    if mode == "oscar-int2":
        assert rot_dir is not None
        cmd += ["--rot-dir", str(rot_dir)]
    if triton_int2 and mode in ("int2", "oscar-int2"):
        cmd += [
            "--prefill-attention-backend",
            "triton",
            "--decode-attention-backend",
            "triton",
        ]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    after = _latest_csv(results_dir)
    if after is None or after in before:
        tail = (proc.stderr or "")[-400:] + (proc.stdout or "")[-400:]
        return False, f"no_new_csv rc={proc.returncode} tail={tail!r}"

    with after.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != mode:
                continue
            ok_raw = (row.get("server_ok") or "").strip().lower()
            ok = ok_raw in ("true", "1", "yes")
            err = (row.get("error") or "").strip()
            return ok, err[:500]
    return False, "mode_row_missing"


def find_max(
    *,
    root: Path,
    bench_sh: Path,
    mode: str,
    results_dir: Path,
    rot_dir: Path | None,
    cap: int,
    max_new_tokens: int,
    health_timeout: float,
    mem_frac: float,
    triton_int2: bool,
    port_offset: int,
) -> tuple[int, str]:
    trial_base = 31000 + port_offset
    trial_seq = 0

    def next_port() -> int:
        nonlocal trial_seq
        trial_seq += 1
        return trial_base + trial_seq * 2

    ok512, err512 = trial(
        root=root,
        bench_sh=bench_sh,
        mode=mode,
        prefill=512,
        results_dir=results_dir,
        rot_dir=rot_dir,
        max_new_tokens=max_new_tokens,
        health_timeout=health_timeout,
        mem_frac=mem_frac,
        triton_int2=triton_int2,
        port=next_port(),
    )
    if not ok512:
        return 0, f"512_failed:{err512}"

    last_ok = 512
    n = 2048
    first_fail: int | None = None
    while n <= cap:
        ok, err = trial(
            root=root,
            bench_sh=bench_sh,
            mode=mode,
            prefill=n,
            results_dir=results_dir,
            rot_dir=rot_dir,
            max_new_tokens=max_new_tokens,
            health_timeout=health_timeout,
            mem_frac=mem_frac,
            triton_int2=triton_int2,
            port=next_port(),
        )
        print(f"    [{mode}] prefill={n} ok={ok} err={err[:120]!r}", flush=True)
        if ok:
            last_ok = n
            n *= 2
            time.sleep(2)
            continue
        first_fail = n
        break

    if first_fail is None:
        return last_ok, "capped_no_fail"

    lo, hi = last_ok, first_fail
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, err = trial(
            root=root,
            bench_sh=bench_sh,
            mode=mode,
            prefill=mid,
            results_dir=results_dir,
            rot_dir=rot_dir,
            max_new_tokens=max_new_tokens,
            health_timeout=health_timeout,
            mem_frac=mem_frac,
            triton_int2=triton_int2,
            port=next_port(),
        )
        if ok:
            lo = mid
        else:
            hi = mid
        time.sleep(2)
    return lo, "binary_search"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rot-dir",
        type=Path,
        default=None,
        help="Required for oscar-int2 (directory containing k/v rotation .pt files).",
    )
    ap.add_argument(
        "--modes",
        default="bf16,int2,oscar-int2",
        help="Comma-separated modes to probe.",
    )
    ap.add_argument("--cap", type=int, default=120_000)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help=(
            "Decode tokens per bench request. Default 1 speeds long-prefill probes; "
            "KV pool sizing still follows bench rules (prefill + max_new + slack)."
        ),
    )
    ap.add_argument("--health-timeout", type=float, default=2400.0)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument(
        "--triton-for-int2-modes",
        action="store_true",
        help="Pass Triton prefill/decode backends for int2 and oscar-int2 (recommended on sm_120).",
    )
    ap.add_argument(
        "--results-subdir",
        default="limit_prefill_probe",
        help="Under repo root results/ directory.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    bench_sh = root / "scripts" / "bench.sh"
    if not bench_sh.is_file():
        print(f"Missing {bench_sh}", file=sys.stderr)
        return 2

    results_dir = root / "results" / args.results_subdir
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary: dict[str, tuple[int, str]] = {}
    for mode in modes:
        rot = args.rot_dir
        if mode == "oscar-int2" and rot is None:
            cand = (
                root
                / "rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
            )
            if cand.is_dir():
                rot = cand
            else:
                print(f"[skip] {mode}: no --rot-dir and default missing {cand}", flush=True)
                continue
        print(f"[probe] mode={mode} cap={args.cap}", flush=True)
        port_offset = (abs(hash(mode)) % 2000) + random.randint(0, 800)
        m, how = find_max(
            root=root,
            bench_sh=bench_sh,
            mode=mode,
            results_dir=results_dir,
            rot_dir=rot if mode == "oscar-int2" else None,
            cap=args.cap,
            max_new_tokens=args.max_new_tokens,
            health_timeout=args.health_timeout,
            mem_frac=args.mem_fraction_static,
            triton_int2=args.triton_for_int2_modes,
            port_offset=port_offset,
        )
        summary[mode] = (m, how)
        print(f"[probe] mode={mode} max_prefill={m} ({how})", flush=True)
        time.sleep(5)

    print("\n=== summary (max prefill tokens, max_new_tokens=%d) ===" % args.max_new_tokens)
    for mode, (m, how) in summary.items():
        print(f"  {mode}: {m}  [{how}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
