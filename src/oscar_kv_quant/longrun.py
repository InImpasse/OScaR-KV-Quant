"""Long-run stability harness for OSCAR mixed INT2 serving."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oscar_kv_quant.paths import to_repo_relative
from oscar_kv_quant.bench import (
    MODE_KV,
    PRESET_TOKENS,
    _build_prefill_text,
    _cuda_graph_enabled,
    _default_max_total_tokens,
    _dtype_size_bytes,
    _gpu_memory_used_mib,
    _int2_group_size_for_head_dim,
    _mode_selected_kv_gib,
    _oscar_env,
    _poll_nvidia_smi_mib,
    _resolve_recent_bf16_tokens,
    _read_tail,
    _server_cmd,
    _validate_rotation_files,
    _wait_health,
    _wait_port_closed,
)
from oscar_kv_quant.kv_estimate import kv_bytes_bf16
from oscar_kv_quant.log_metrics import parse_server_log
from oscar_kv_quant.profiles import PROFILES, resolve_kv_geometry


ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError:",
    "AssertionError:",
    "CUDA out of memory",
    "out of memory",
    "HP-prefix pool exhausted",
    "Scheduler hit an exception",
    "SIGQUIT received",
)


@dataclass
class RequestResult:
    index: int
    ok: bool
    elapsed_sec: float
    completion_tokens: int
    tok_s: float
    error: str


def _extract_tokens(data: dict[str, Any], fallback: int) -> int:
    usage = data.get("usage") or {}
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        if usage.get(key) is not None:
            return int(usage[key])
    if data.get("meta_info", {}).get("completion_tokens") is not None:
        return int(data["meta_info"]["completion_tokens"])
    return fallback


def _run_one_request(
    base: str,
    prompt: str,
    *,
    max_new_tokens: int,
    request_api: str,
    timeout: float,
) -> tuple[int, str]:
    import httpx

    with httpx.Client(timeout=timeout, trust_env=False) as client:
        if request_api == "chat":
            body: dict[str, Any] = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
            r = client.post(f"{base}/v1/chat/completions", json=body)
        elif request_api == "completions":
            body = {
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
            r = client.post(f"{base}/v1/completions", json=body)
        else:
            body = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            }
            r = client.post(f"{base}/generate", json=body)
        r.raise_for_status()
        return _extract_tokens(r.json(), max_new_tokens), ""


def _scan_log_errors(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    hits: list[str] = []
    for line in lines:
        if any(pattern in line for pattern in ERROR_PATTERNS):
            hits.append(line.strip())
    return hits[-20:]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _half_medians(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    split = max(1, len(values) // 2)
    first = values[:split]
    second = values[split:]
    return _median(first), _median(second) if second else None


def _write_report(path: Path, data: dict[str, Any]) -> None:
    rows = data["requests"]
    lines = [
        "# OSCAR-KV-Quant longrun",
        "",
        f"Generated: {data['generated']}",
        f"Mode: `{data['mode']}`",
        f"Model path: `{data['model_path']}`",
        f"Prompt tokens: `{data['prefill_tokens']}`",
        f"Max new tokens: `{data['max_new_tokens']}`",
        f"Requests: `{data['num_requests']}`",
        "",
        "## Summary",
        "",
        f"- ok: `{data['ok']}`",
        f"- failures: `{data['failures']}`",
        f"- request tok/s median: `{data['request_tok_s_median']}`",
        f"- request tok/s p05: `{data['request_tok_s_p05']}`",
        f"- peak MiB: `{data['peak_mib_total']}`",
        f"- KV K/V GB: `{data['kv_k_size_gb']}` / `{data['kv_v_size_gb']}`",
        f"- HP-prefix pool tokens: `{data['hp_prefix_pool_tokens']}`",
        f"- log errors: `{len(data['log_errors'])}`",
        "",
        "## Requests",
        "",
        "| i | ok | elapsed s | completion tokens | tok/s | error |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {index} | {ok} | {elapsed_sec:.3f} | {completion_tokens} | "
            "{tok_s:.3f} | `{error}` |".format(**row)
        )
    if data["log_errors"]:
        lines += ["", "## Log Errors", ""]
        lines.extend(f"- `{line[:240]}`" for line in data["log_errors"])
    path.write_text("\n".join(lines) + "\n")


def iter_modes(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run sequential longrun stability checks.")
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="granite")
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--preset", choices=sorted(PRESET_TOKENS.keys()), default="long")
    ap.add_argument("--prefill-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--mode", choices=sorted(MODE_KV.keys()), default="oscar-int2")
    ap.add_argument("--rot-dir", type=Path, default=None)
    ap.add_argument("--k-rotation-filename", default="k_rotation_qqt_r_h_pbr.pt")
    ap.add_argument("--v-rotation-filename", default="v_rotation_sst_r_h_pbr.pt")
    ap.add_argument("--oscar-k-clip-ratio", type=float, default=0.96)
    ap.add_argument("--oscar-v-clip-ratio", type=float, default=0.92)
    ap.add_argument("--lloyd-max", type=int, default=0)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the SGLang server subprocess.",
    )
    ap.add_argument("--port", type=int, default=31988)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-total-tokens", type=int, default=None)
    ap.add_argument("--request-api", choices=["chat", "completions", "generate"], default="completions")
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument("--results-dir", type=Path, default=Path("results/longrun"))
    ap.add_argument("--prefill-attention-backend", default=None)
    ap.add_argument("--decode-attention-backend", default=None)
    ap.add_argument("--chunked-prefill-size", type=int, default=None)
    ap.add_argument("--max-running-requests", type=int, default=1)
    ap.add_argument("--max-queued-requests", type=int, default=8)
    ap.add_argument("--prefix-bf16-tokens", type=int, default=64)
    ap.add_argument(
        "--recent-bf16-tokens",
        default="auto",
        help=(
            "OSCAR mixed-KV high-precision recent tokens; integer or 'auto'. "
            "auto uses 64 for <=64 generated tokens, otherwise 256."
        ),
    )
    ap.add_argument("--hp-prefix-pool-tokens", type=int, default=None)
    ap.add_argument("--mixed-kv-max-quant-tokens", type=int, default=None)
    ap.add_argument("--mixed-kv-hp-max-splits", type=int, default=4)
    ap.add_argument("--mixed-kv-scale-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--enable-cuda-graph", action="store_true")
    ap.add_argument("--disable-oscar-cuda-graph", action="store_true")
    ap.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile = PROFILES[args.profile]
    model_path = Path(args.model_path or profile.default_model_path)
    geometry = resolve_kv_geometry(model_path, profile)
    int2_group_size = _int2_group_size_for_head_dim(geometry.head_dim)
    prefill_n = args.prefill_tokens or PRESET_TOKENS[args.preset]
    seq = prefill_n + args.max_new_tokens
    recent_bf16_tokens = _resolve_recent_bf16_tokens(
        args.recent_bf16_tokens, max_new_tokens=args.max_new_tokens
    )
    mode = args.mode
    kv = MODE_KV[mode]
    max_total_tokens = (
        args.max_total_tokens
        if args.max_total_tokens is not None
        else _default_max_total_tokens(
            mode,
            prefill_tokens=prefill_n,
            max_new_tokens=args.max_new_tokens,
            max_running_requests=args.max_running_requests,
        )
    )
    mixed_kv_max_quant_tokens = None
    env = os.environ.copy()
    if mode == "oscar-int2":
        valid, msg = _validate_rotation_files(
            args.rot_dir,
            mode,
            k_rotation_filename=args.k_rotation_filename,
            v_rotation_filename=args.v_rotation_filename,
        )
        if not valid:
            raise SystemExit(msg)
        assert args.rot_dir is not None
        mixed_kv_max_quant_tokens = (
            args.mixed_kv_max_quant_tokens
            if args.mixed_kv_max_quant_tokens is not None
            else max_total_tokens
        )
        env.update(
            _oscar_env(
                args.rot_dir,
                args.prefix_bf16_tokens,
                recent_bf16_tokens,
                args.hp_prefix_pool_tokens,
                max_quant_tokens=mixed_kv_max_quant_tokens,
                hp_max_splits=args.mixed_kv_hp_max_splits,
                scale_dtype=args.mixed_kv_scale_dtype,
                fused_rotate_clip_quant=True,
                k_rotation_filename=args.k_rotation_filename,
                v_rotation_filename=args.v_rotation_filename,
                k_clip_ratio=args.oscar_k_clip_ratio,
                v_clip_ratio=args.oscar_v_clip_ratio,
                lloyd_max=args.lloyd_max,
            )
        )

    cmd = _server_cmd(
        args.python,
        model_path,
        args.port,
        args.port + 1000,
        kv,
        mode,
        args.mem_fraction_static,
        max_total_tokens,
        trust_remote=not args.no_trust_remote_code,
        prefill_backend=args.prefill_attention_backend,
        decode_backend=args.decode_attention_backend,
        json_model_override_args=None,
        max_running_requests=args.max_running_requests,
        max_queued_requests=args.max_queued_requests,
        disable_cuda_graph=not _cuda_graph_enabled(args, mode),
        disable_piecewise_cuda_graph=not args.enable_piecewise_cuda_graph,
        piecewise_cuda_graph_max_tokens=None,
        chunked_prefill_size=args.chunked_prefill_size,
        enable_memory_saver=False,
        int2_group_size=int2_group_size,
    )

    selected_gib = _mode_selected_kv_gib(
        mode,
        geometry,
        seq,
        args.prefix_bf16_tokens,
        recent_bf16_tokens,
        pool_tokens=mixed_kv_max_quant_tokens,
        max_running_requests=args.max_running_requests,
        hp_prefix_pool_tokens=args.hp_prefix_pool_tokens,
        group_size=int2_group_size,
        scale_dtype_bytes=_dtype_size_bytes(args.mixed_kv_scale_dtype),
    )
    bf16_gib = kv_bytes_bf16(
        geometry.layers_for_kv_estimate,
        geometry.num_kv_heads,
        seq,
        geometry.head_dim,
    ) / (1024**3)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.results_dir / f"{args.profile}_{mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    json_path = out_dir / "longrun.json"
    md_path = out_dir / "longrun.md"

    if args.dry_run:
        print("[longrun:dry-run] cmd=" + " ".join(cmd), flush=True)
        if mode.startswith("oscar-"):
            relevant = {k: env[k] for k in sorted(env) if k.startswith("SGLANG_")}
            print("[longrun:dry-run] OSCAR env=" + json.dumps(relevant, indent=2), flush=True)
        print(f"[longrun:dry-run] selected_kv_gib={selected_gib:.4f} bf16_kv_gib={bf16_gib:.4f}", flush=True)
        return

    prompt = _build_prefill_text(model_path, prefill_n)
    base = f"http://127.0.0.1:{args.port}"
    baseline = _gpu_memory_used_mib()
    samples: list[float] = []
    stop_poll = threading.Event()
    poller = threading.Thread(target=_poll_nvidia_smi_mib, args=(samples, stop_poll))
    poller.start()
    proc: subprocess.Popen[str] | None = None
    results: list[RequestResult] = []
    try:
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
        print(f"[longrun] wait health mode={mode}", flush=True)
        if not _wait_health(base, proc, args.health_timeout):
            raise RuntimeError("server_health_timeout: " + _read_tail(log_path, 20))
        print(f"[longrun] start requests mode={mode} n={args.num_requests}", flush=True)
        for i in range(1, args.num_requests + 1):
            if proc.poll() is not None:
                results.append(RequestResult(i, False, 0.0, 0, 0.0, "server_exited"))
                break
            t0 = time.perf_counter()
            try:
                tokens, err = _run_one_request(
                    base,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    request_api=args.request_api,
                    timeout=args.request_timeout,
                )
                elapsed = time.perf_counter() - t0
                tok_s = tokens / elapsed if elapsed > 0 else 0.0
                ok = err == ""
                results.append(RequestResult(i, ok, elapsed, tokens, tok_s, err))
                print(f"[longrun] request {i}/{args.num_requests} ok={ok} tok/s={tok_s:.2f} err={err!r}", flush=True)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                results.append(RequestResult(i, False, elapsed, 0, 0.0, str(exc)))
                print(f"[longrun] request {i}/{args.num_requests} ok=False err={exc!s}", flush=True)
                if proc.poll() is not None:
                    break
    finally:
        stop_poll.set()
        poller.join(timeout=5)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        _wait_port_closed(args.port)

    metrics = parse_server_log(
        log_path,
        measurement_requests=max(1, len(results)),
        decode_tokens_per_request=args.max_new_tokens,
    )
    log_errors = _scan_log_errors(log_path)
    failures = [r for r in results if not r.ok]
    tok_s_vals = [r.tok_s for r in results if r.ok]
    request_first_half_median, request_second_half_median = _half_medians(tok_s_vals)
    request_log_windows = [asdict(w) for w in metrics.request_windows[-len(results) :]]
    cached_prefill_tok_s_vals = [
        float(w["prefill_tok_s"])
        for w in request_log_windows
        if (w.get("prefill_cached_tokens") or 0) > 0 and w.get("prefill_tok_s") is not None
    ]
    decode_first_tok_s_vals = [
        float(w["decode_first_tok_s"])
        for w in request_log_windows
        if w.get("decode_first_tok_s") is not None
    ]
    cached_prefill_wall_vals = [
        float(w["prefill_wall_est_s"])
        for w in request_log_windows
        if (w.get("prefill_cached_tokens") or 0) > 0
        and w.get("prefill_wall_est_s") is not None
    ]
    cached_prefill_interval_vals = [
        float(w["prefill_interval_est_s"])
        for w in request_log_windows
        if (w.get("prefill_cached_tokens") or 0) > 0
        and w.get("prefill_interval_est_s") is not None
    ]
    peak = max(samples) if samples else None
    data: dict[str, Any] = {
        "generated": ts,
        "profile": args.profile,
        "mode": mode,
        "model_path": to_repo_relative(model_path),
        "prefill_tokens": prefill_n,
        "max_new_tokens": args.max_new_tokens,
        "num_requests": args.num_requests,
        "completed_requests": len(results),
        "failures": len(failures),
        "ok": len(results) == args.num_requests and not failures and not log_errors,
        "baseline_mib": baseline,
        "peak_mib_total": peak,
        "peak_mib_delta": None if peak is None or baseline is None else max(0.0, peak - baseline),
        "request_tok_s_median": _median(tok_s_vals),
        "request_tok_s_first_half_median": request_first_half_median,
        "request_tok_s_second_half_median": request_second_half_median,
        "request_tok_s_min": min(tok_s_vals) if tok_s_vals else None,
        "request_tok_s_p05": _percentile(tok_s_vals, 5),
        "server_decode_steady_median_tok_s": metrics.decode_steady_median_tok_s,
        "server_decode_max_tok_s": metrics.decode_max_tok_s,
        "prefill_median_tok_s": metrics.prefill_median_tok_s,
        "cached_prefill_tok_s_median": _median(cached_prefill_tok_s_vals),
        "cached_prefill_tok_s_min": (
            min(cached_prefill_tok_s_vals) if cached_prefill_tok_s_vals else None
        ),
        "cached_prefill_tok_s_p05": _percentile(cached_prefill_tok_s_vals, 5),
        "cached_prefill_wall_est_s_median": _median(cached_prefill_wall_vals),
        "cached_prefill_wall_est_s_max": (
            max(cached_prefill_wall_vals) if cached_prefill_wall_vals else None
        ),
        "cached_prefill_wall_est_s_p95": _percentile(cached_prefill_wall_vals, 95),
        "cached_prefill_interval_est_s_median": _median(cached_prefill_interval_vals),
        "cached_prefill_interval_est_s_max": (
            max(cached_prefill_interval_vals) if cached_prefill_interval_vals else None
        ),
        "cached_prefill_interval_est_s_p95": _percentile(
            cached_prefill_interval_vals, 95
        ),
        "cached_prefill_new_median_tokens": metrics.cached_prefill_new_median_tokens,
        "cached_prefill_cached_median_tokens": metrics.cached_prefill_cached_median_tokens,
        "cached_prefill_cache_ratio_median": metrics.cached_prefill_cache_ratio_median,
        "decode_first_tok_s_median": _median(decode_first_tok_s_vals),
        "decode_first_tok_s_min": (
            min(decode_first_tok_s_vals) if decode_first_tok_s_vals else None
        ),
        "kv_pool_tokens": metrics.kv_pool_tokens,
        "kv_k_size_gb": metrics.kv_k_size_gb,
        "kv_v_size_gb": metrics.kv_v_size_gb,
        "hp_prefix_pool_tokens": metrics.hp_prefix_pool_tokens,
        "unified_mixed_kv": metrics.unified_mixed_kv,
        "selected_kv_gib": selected_gib,
        "bf16_kv_gib": bf16_gib,
        "selected_vs_bf16_kv_ratio": selected_gib / bf16_gib if bf16_gib else None,
        "server_log_path": str(log_path),
        "log_tail": _read_tail(log_path, 80),
        "log_errors": log_errors,
        "requests": [asdict(r) for r in results],
        "request_log_windows": request_log_windows,
    }
    json_path.write_text(json.dumps(data, indent=2))
    _write_report(md_path, data)
    print(f"[longrun] wrote {json_path} and {md_path}", flush=True)
    if not data["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
