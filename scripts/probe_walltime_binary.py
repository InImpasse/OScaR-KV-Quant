#!/usr/bin/env python3
"""Binary-search max prefill tokens under a wall-clock threshold (oscar-kv-bench trial).

Matches README RTX 5050 long-prefill methodology:
  granite profile, max_new_tokens=1, mem_fraction_static=0.88,
  --enable-cuda-graph for all modes, Triton prefill/decode for int2/oscar-int2,
  completions API, warmup 0 / bench 1.

Pass iff CSV reports server_ok for the mode AND trial wall (perf_counter around
bench subprocess) <= threshold_sec.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
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
    trial_timeout: float | None,
) -> tuple[bool, float, dict[str, str]]:
    """Return (pass_under_threshold, elapsed_s, detail_dict)."""
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
        "--enable-cuda-graph",
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
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    t0 = time.perf_counter()
    try:
        stdout, stderr = proc.communicate(timeout=trial_timeout)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - t0
        return (
            False,
            elapsed,
            {
                "reason": "trial_timeout",
                "tail": ((stderr or "")[-400:] + (stdout or "")[-400:]).replace("\n", "\\n"),
            },
        )

    after = _latest_csv(results_dir)
    detail: dict[str, str] = {"rc": str(proc.returncode)}
    if after is None or after in before:
        tail = (stderr or "")[-400:] + (stdout or "")[-400:]
        detail["reason"] = "no_new_csv"
        detail["tail"] = tail.replace("\n", "\\n")
        return False, elapsed, detail

    ok = False
    err = ""
    with after.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != mode:
                continue
            ok_raw = (row.get("server_ok") or "").strip().lower()
            ok = ok_raw in ("true", "1", "yes")
            err = (row.get("error") or "").strip()
            break
    detail["server_ok"] = str(ok)
    detail["error"] = err[:500]
    detail["csv"] = str(after)
    if not ok:
        detail["reason"] = "server_not_ok"
        return False, elapsed, detail
    return True, elapsed, detail


def passes(
    *,
    threshold: float,
    trial_timeout: float,
    ok_elapsed: tuple[bool, float, dict[str, str]],
) -> bool:
    ok, elapsed, d = ok_elapsed
    if not ok:
        return False
    if elapsed > threshold + 1e-6:
        d["reason"] = f"wall_gt_threshold elapsed={elapsed:.3f}s > {threshold}s"
        return False
    return True


def binary_search_aligned(
    *,
    root: Path,
    bench_sh: Path,
    mode: str,
    threshold: float,
    lo_tokens: int,
    hi_tokens_fail: int,
    step: int,
    results_dir: Path,
    rot_dir: Path | None,
    max_new_tokens: int,
    health_timeout: float,
    mem_frac: float,
    triton_int2: bool,
    port_base: int,
    trial_timeout: float,
    sleep_s: float,
) -> dict:
    """Assume passes(lo) and not passes(hi_tokens_fail); both divisible by step."""
    assert lo_tokens % step == 0 and hi_tokens_fail % step == 0
    lo_i = lo_tokens // step
    hi_i = hi_tokens_fail // step
    assert hi_i > lo_i
    trials: list[dict] = []

    def run(prefill: int) -> tuple[bool, float, dict]:
        nonlocal port_base
        port_base += 2
        return trial(
            root=root,
            bench_sh=bench_sh,
            mode=mode,
            prefill=prefill,
            results_dir=results_dir,
            rot_dir=rot_dir,
            max_new_tokens=max_new_tokens,
            health_timeout=health_timeout,
            mem_frac=mem_frac,
            triton_int2=triton_int2,
            port=port_base,
            trial_timeout=trial_timeout,
        )

    # Anchor endpoints (caller already validated; re-check for safety).
    r_lo = run(lo_tokens)
    trials.append({"prefill": lo_tokens, "ok": passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r_lo), "elapsed_s": r_lo[1], **r_lo[2]})
    time.sleep(sleep_s)
    r_hi = run(hi_tokens_fail)
    trials.append(
        {
            "prefill": hi_tokens_fail,
            "ok": passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r_hi),
            "elapsed_s": r_hi[1],
            **r_hi[2],
        }
    )
    time.sleep(sleep_s)
    if not trials[0]["ok"]:
        return {"mode": mode, "threshold_s": threshold, "max_prefill": None, "error": "lo_endpoint_not_passing", "trials": trials}
    if trials[1]["ok"]:
        return {"mode": mode, "threshold_s": threshold, "max_prefill": None, "error": "hi_endpoint_still_passing", "trials": trials}

    while hi_i - lo_i > 1:
        mid_i = (lo_i + hi_i) // 2
        mid = mid_i * step
        r = run(mid)
        ok_mid = passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r)
        trials.append({"prefill": mid, "ok": ok_mid, "elapsed_s": r[1], **r[2]})
        if ok_mid:
            lo_i = mid_i
        else:
            hi_i = mid_i
        time.sleep(sleep_s)

    max_prefill = lo_i * step
    return {"mode": mode, "threshold_s": threshold, "max_prefill": max_prefill, "trials": trials}


def binary_search_dense(
    *,
    root: Path,
    bench_sh: Path,
    mode: str,
    threshold: float,
    lo_tokens: int,
    hi_tokens_fail: int,
    results_dir: Path,
    rot_dir: Path | None,
    max_new_tokens: int,
    health_timeout: float,
    mem_frac: float,
    triton_int2: bool,
    port_base: int,
    trial_timeout: float,
    sleep_s: float,
) -> dict:
    lo, hi = lo_tokens, hi_tokens_fail
    trials: list[dict] = []

    def run(prefill: int) -> tuple[bool, float, dict]:
        nonlocal port_base
        port_base += 2
        return trial(
            root=root,
            bench_sh=bench_sh,
            mode=mode,
            prefill=prefill,
            results_dir=results_dir,
            rot_dir=rot_dir,
            max_new_tokens=max_new_tokens,
            health_timeout=health_timeout,
            mem_frac=mem_frac,
            triton_int2=triton_int2,
            port=port_base,
            trial_timeout=trial_timeout,
        )

    r_lo = run(lo)
    trials.append({"prefill": lo, "ok": passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r_lo), "elapsed_s": r_lo[1], **r_lo[2]})
    time.sleep(sleep_s)
    r_hi = run(hi)
    trials.append({"prefill": hi, "ok": passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r_hi), "elapsed_s": r_hi[1], **r_hi[2]})
    time.sleep(sleep_s)
    if not trials[0]["ok"]:
        return {"mode": mode, "threshold_s": threshold, "max_prefill": None, "error": "lo_endpoint_not_passing", "trials": trials}
    if trials[1]["ok"]:
        return {"mode": mode, "threshold_s": threshold, "max_prefill": None, "error": "hi_endpoint_still_passing", "trials": trials}

    while lo + 1 < hi:
        mid = (lo + hi + 1) // 2
        r = run(mid)
        ok_mid = passes(threshold=threshold, trial_timeout=trial_timeout, ok_elapsed=r)
        trials.append({"prefill": mid, "ok": ok_mid, "elapsed_s": r[1], **r[2]})
        if ok_mid:
            lo = mid
        else:
            hi = mid
        time.sleep(sleep_s)

    return {"mode": mode, "threshold_s": threshold, "max_prefill": lo, "trials": trials}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rot-dir",
        type=Path,
        default=None,
        help="oscar-int2 rotation directory (default: GPQA rotations if present).",
    )
    ap.add_argument(
        "--results-subdir",
        default="probe_walltime_binary_granite",
        help="results/<subdir> for CSVs and summary.jsonl",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--health-timeout", type=float, default=2400.0)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--sleep-between-trials", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    bench_sh = root / "scripts" / "bench.sh"
    if not bench_sh.is_file():
        print(f"Missing {bench_sh}", file=sys.stderr)
        return 2

    rot = args.rot_dir
    if rot is None:
        cand = root / "rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
        rot = cand if cand.is_dir() else None

    results_dir = root / "results" / args.results_subdir
    summary_path = results_dir / "summary.jsonl"
    _ensure_results_parent(results_dir)

    # README brackets (CUDA graph on, Triton int2): lo passes / hi fails threshold.
    jobs: list[tuple[str, float, int, int, int]] = [
        # mode, threshold_sec, lo_good, hi_fail, step (1 => use dense binary)
        ("bf16", 360.0, 41952, 41968, 1),
        ("bf16", 600.0, 41952, 41968, 1),
        ("int2", 360.0, 65536, 69632, 128),
        ("int2", 600.0, 80896, 81920, 128),
        ("oscar-int2", 360.0, 65536, 69632, 128),
        ("oscar-int2", 600.0, 80896, 81920, 128),
    ]

    trial_timeout = 900.0  # allow ~606s 81920 runs + margin

    done_keys: set[tuple[str, float]] = set()
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done_keys.add((rec["mode"], float(rec["threshold_s"])))

    for mode, thr, lo, hi, step in jobs:
        if (mode, thr) in done_keys:
            print(f"[skip] already done: mode={mode} threshold_s={thr}", flush=True)
            continue
        port_base = 32000 + (abs(hash(f"{mode}-{thr}")) % 1500) + random.randint(0, 400)
        if mode == "oscar-int2" and rot is None:
            rec = {"mode": mode, "threshold_s": thr, "max_prefill": None, "error": "missing_rot_dir"}
            summary_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(rec, flush=True)
            continue

        sub = results_dir / f"{mode}_thr{int(thr)}"
        _ensure_results_parent(sub)
        if step == 1:
            out = binary_search_dense(
                root=root,
                bench_sh=bench_sh,
                mode=mode,
                threshold=thr,
                lo_tokens=lo,
                hi_tokens_fail=hi,
                results_dir=sub,
                rot_dir=rot,
                max_new_tokens=args.max_new_tokens,
                health_timeout=args.health_timeout,
                mem_frac=args.mem_fraction_static,
                triton_int2=True,
                port_base=port_base,
                trial_timeout=trial_timeout,
                sleep_s=args.sleep_between_trials,
            )
        else:
            out = binary_search_aligned(
                root=root,
                bench_sh=bench_sh,
                mode=mode,
                threshold=thr,
                lo_tokens=lo,
                hi_tokens_fail=hi,
                step=step,
                results_dir=sub,
                rot_dir=rot,
                max_new_tokens=args.max_new_tokens,
                health_timeout=args.health_timeout,
                mem_frac=args.mem_fraction_static,
                triton_int2=True,
                port_base=port_base,
                trial_timeout=trial_timeout,
                sleep_s=args.sleep_between_trials,
            )
        with summary_path.open("a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(json.dumps(out, ensure_ascii=False), flush=True)
        time.sleep(5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
