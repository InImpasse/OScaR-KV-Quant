"""Run SGLang server per KV mode and measure decode tok/s + GPU memory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oscar_kv_quant.kv_estimate import (
    fmt_gib,
    kv_bytes_bf16,
    kv_bytes_int2_packed_naive,
    kv_bytes_oscar_mixed_runtime_estimate,
)
from oscar_kv_quant.log_metrics import ServerLogMetrics, parse_server_log
from oscar_kv_quant.paths import to_repo_relative
from oscar_kv_quant.profiles import PROFILES, KVGeometry, load_model_config, resolve_kv_geometry

PRESET_TOKENS = {
    "short": 512,
    "medium": 2048,
    "long": 8192,
    "16k": 16384,
    "32k": 32768,
}

# Logical names -> sglang --kv-cache-dtype (see README KV modes table)
MODE_KV = {
    "bf16": "bf16",
    "fp16": "auto",
    "fp8": "fp8_e4m3",
    "fp4": "fp4_e2m1",
    "int8": "int8",
    "int4": "int4",
    "int2": "int2",
    "oscar-int2": "int2",
    "oscar-int8": "int8",
    "oscar-int4": "int4",
}

MODE_NOTES = {
    "bf16": "BF16 KV cache baseline",
    "fp16": "FP16-style baseline (--kv-cache-dtype auto; bf16 weights on Granite muP models)",
    "fp8": "8-bit floating KV cache (fp8_e4m3)",
    "int8": "Symmetric int8 KV + internal bf16 shadow for Triton (true integer storage)",
    "fp4": "4-bit floating KV cache (fp4_e2m1 / MXFP4), hardware dependent",
    "int4": "Symmetric int4 (nibble-packed) KV + bf16 shadow for Triton; head_dim must be even",
    "int2": "SGLang Triton INT2 KV cache without OSCAR rotations",
    "oscar-int2": "OSCAR INT2 mixed KV windows with rotation files",
    "oscar-int8": "int8 KV + Oscar (SGLANG_OSCAR_ROTATE_QUANT_KV + rotation checkpoints)",
    "oscar-int4": "int4 KV + Oscar (SGLANG_OSCAR_ROTATE_QUANT_KV + rotation checkpoints)",
}


def _short_context_oscar_fallback_mode(
    mode: str,
    prefill_tokens: int,
    min_prefill_tokens: int,
    fallback: str,
) -> str | None:
    if mode != "oscar-int2":
        return None
    if min_prefill_tokens <= 0 or prefill_tokens >= min_prefill_tokens:
        return None
    return fallback


@dataclass
class BenchRow:
    profile: str
    mode: str
    kv_dtype_cli: str
    kv_mode_note: str
    request_api: str
    prefill_tokens: int
    max_new_tokens: int
    decode_toks_per_sec: float
    request_toks_per_sec: float
    decode_log_toks_per_sec: float | None
    baseline_mib: float | None
    peak_mib_total: float | None
    peak_mib_delta: float | None
    server_pid: int | None
    server_log_path: str
    server_ok: bool
    error: str
    log_tail: str
    num_layers: int
    num_attention_layers: int | None
    num_kv_heads: int
    head_dim: int
    kv_theory_bf16_gib: float
    kv_theory_selected_gib: float
    decode_first_tok_s: float | None
    decode_steady_median_tok_s: float | None
    decode_steady_p95_tok_s: float | None
    decode_flush_median_tok_s: float | None
    prefill_median_tok_s: float | None
    prefill_wall_s: float | None
    decode_wall_s: float | None
    flush_step_fraction: float | None
    effective_decode_tok_s: float | None
    kv_pool_tokens: int | None
    kv_k_size_gb: float | None
    kv_v_size_gb: float | None
    cuda_graph_enabled: bool
    profile_dir: str


def _gpu_memory_used_mib() -> float | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def _poll_nvidia_smi_mib(samples: list[float], stop: threading.Event) -> None:
    while not stop.is_set():
        v = _gpu_memory_used_mib()
        if v is not None:
            samples.append(v)
        time.sleep(0.25)


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_port_closed(port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _port_is_open("127.0.0.1", port):
            return True
        time.sleep(0.5)
    return False


def _read_tail(path: Path, n_lines: int = 60) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def _build_prefill_text(model_path: Path, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    local = model_path.is_dir() and (model_path / "config.json").is_file()
    tok = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=local
    )
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tok(seed, add_special_tokens=False).input_ids
    if not ids:
        ids = [tok.eos_token_id or 0]
    out_ids: list[int] = []
    while len(out_ids) < target_tokens:
        out_ids.extend(ids)
    out_ids = out_ids[:target_tokens]
    return tok.decode(out_ids, skip_special_tokens=True)


def _rotation_paths(
    rot_dir: Path,
    *,
    k_rotation_filename: str = "k_rotation_qqt_r_h_pbr.pt",
    v_rotation_filename: str = "v_rotation_sst_r_h_pbr.pt",
) -> tuple[Path, Path]:
    return (
        rot_dir / k_rotation_filename,
        rot_dir / v_rotation_filename,
    )


def _validate_rotation_files(
    rot_dir: Path | None,
    mode_label: str = "oscar",
    *,
    k_rotation_filename: str = "k_rotation_qqt_r_h_pbr.pt",
    v_rotation_filename: str = "v_rotation_sst_r_h_pbr.pt",
) -> tuple[bool, str]:
    if rot_dir is None:
        return False, f"{mode_label} requires --rot-dir"
    k, v = _rotation_paths(
        rot_dir,
        k_rotation_filename=k_rotation_filename,
        v_rotation_filename=v_rotation_filename,
    )
    missing = [str(p) for p in (k, v) if not p.is_file()]
    if missing:
        return False, "missing rotation file(s): " + ", ".join(missing)
    return True, ""


def _is_cuda_sm120() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
    except Exception:
        return False


def _cuda_graph_enabled(args: argparse.Namespace, mode: str) -> bool:
    """OSCAR mixed-KV decode is much faster with CUDA graphs; enable by default."""
    if getattr(args, "disable_oscar_cuda_graph", False) and mode.startswith("oscar-"):
        return False
    if args.enable_cuda_graph:
        return True
    return mode.startswith("oscar-")


def _default_prefill_backend(mode: str) -> str:
    if mode in ("int2", "oscar-int2"):
        # SM120 consumer Blackwell cannot run the current FA3/FA4 paths used by
        # this INT2 dense-prefill code. Triton stays on the quantized KV path
        # and falls back to SDPA only for the incompatible dense prefill step.
        return "triton" if _is_cuda_sm120() else "fa3"
    return "triton"


def _oscar_quant_kv_env(
    rot_dir: Path,
    *,
    k_rotation_filename: str = "k_rotation_qqt_r_h_pbr.pt",
    v_rotation_filename: str = "v_rotation_sst_r_h_pbr.pt",
) -> dict[str, str]:
    """Env for FP8/FP4 KV + learned Oscar rotations (plain MHA pool, not mixed int2)."""
    k, v = _rotation_paths(
        rot_dir,
        k_rotation_filename=k_rotation_filename,
        v_rotation_filename=v_rotation_filename,
    )
    return {
        "SGLANG_OSCAR_ROTATE_QUANT_KV": "1",
        "SGLANG_OSCAR_ABSORB_V_ROTATION": "1",
        "SGLANG_OSCAR_K_ROTATION_PATH": str(k),
        "SGLANG_OSCAR_V_ROTATION_PATH": str(v),
    }


def _oscar_env(
    rot_dir: Path,
    prefix_tokens: int,
    recent_tokens: int,
    hp_prefix_pool_tokens: int | None,
    max_quant_tokens: int | None = None,
    hp_max_splits: int = 4,
    scale_dtype: str = "bfloat16",
    fused_rotate_clip_quant: bool = True,
    k_rotation_filename: str = "k_rotation_qqt_r_h_pbr.pt",
    v_rotation_filename: str = "v_rotation_sst_r_h_pbr.pt",
    k_clip_ratio: float = 0.96,
    v_clip_ratio: float = 0.92,
    lloyd_max: int = 0,
) -> dict[str, str]:
    k, v = _rotation_paths(
        rot_dir,
        k_rotation_filename=k_rotation_filename,
        v_rotation_filename=v_rotation_filename,
    )
    env = {
        "SGLANG_ENABLE_MIXED_KV_WINDOWS": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_OSCAR_ABSORB_V_ROTATION": "1",
        "SGLANG_MIXED_KV_HP_MAX_SPLITS": str(hp_max_splits),
        "SGLANG_MIXED_KV_PREFIX_TOKENS": str(prefix_tokens),
        "SGLANG_MIXED_KV_RECENT_TOKENS": str(recent_tokens),
        "SGLANG_MIXED_KV_HP_DTYPE": "bfloat16",
        "SGLANG_MIXED_KV_SCALE_DTYPE": scale_dtype,
        "SGLANG_MIXED_KV_MAX_QUANT_TOKENS": str(max_quant_tokens or 0),
        "SGLANG_OSCAR_K_ROTATION_PATH": str(k),
        "SGLANG_OSCAR_V_ROTATION_PATH": str(v),
        "SGLANG_OSCAR_K_CLIP_RATIO": str(k_clip_ratio),
        "SGLANG_OSCAR_V_CLIP_RATIO": str(v_clip_ratio),
        "SGLANG_LLOYD_MAX": str(lloyd_max),
        "SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT": "1" if fused_rotate_clip_quant else "0",
    }
    if hp_prefix_pool_tokens is not None:
        env["SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS"] = str(hp_prefix_pool_tokens)
    return env


def _default_max_total_tokens(
    mode: str,
    *,
    prefill_tokens: int,
    max_new_tokens: int,
    max_running_requests: int,
) -> int | None:
    """Choose a fair KV-pool cap when the caller does not provide one.

    For single-request long-context comparisons, SGLang's auto-sized pool can
    let int2/oscar modes consume far more logical KV slots than bf16, which
    hides the memory advantage we actually want to measure. Cap the pool to the
    active request budget plus a small page-aligned slack so bf16 and oscar are
    compared at similar effective capacity.
    """
    if max_running_requests != 1:
        return None
    if mode not in ("int2", "oscar-int2", "bf16"):
        return None
    # Empirically stable on RTX 5050 single-request Granite runs:
    # keep only a small slack over the active request budget so oscar/int2
    # do not auto-allocate a much larger logical pool than bf16.
    target = prefill_tokens + max_new_tokens + 896
    page = 8 if mode == "oscar-int2" else 128
    return ((target + page - 1) // page) * page


def _int2_group_size_for_head_dim(head_dim: int) -> int:
    preferred = 128
    if head_dim % preferred == 0:
        return preferred
    return head_dim


def _resolve_recent_bf16_tokens(value: str | int, *, max_new_tokens: int) -> int:
    if isinstance(value, int):
        return value
    if value == "auto":
        return 64 if max_new_tokens <= 64 else 256
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--recent-bf16-tokens must be an integer or 'auto'"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("--recent-bf16-tokens must be >= 0")
    return parsed


def _fp16_weights_unsafe_for_model(model_path: Path) -> bool:
    """Granite muP models underflow in float16 and emit degenerate outputs."""
    cfg = load_model_config(model_path)
    model_type = str(cfg.get("model_type", "")).lower()
    torch_dtype = str(cfg.get("torch_dtype", "")).lower()
    return model_type in {"granite", "granitemoe", "granitemoehybrid"} or (
        torch_dtype in {"bfloat16", "bf16"} and "granite" in model_type
    )


def _server_cmd(
    py: str,
    model_path: Path,
    port: int,
    dist_port: int,
    kv_dtype: str,
    mode: str,
    mem_frac: float,
    max_total_tokens: int | None,
    trust_remote: bool,
    prefill_backend: str | None,
    decode_backend: str | None,
    json_model_override_args: str | None,
    max_running_requests: int,
    max_queued_requests: int,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    piecewise_cuda_graph_max_tokens: int | None = None,
    chunked_prefill_size: int | None = None,
    enable_memory_saver: bool = False,
    int2_group_size: int = 128,
) -> list[str]:
    prefill_be = prefill_backend or _default_prefill_backend(mode)
    decode_be = decode_backend or "triton"
    page_size = "8" if mode == "oscar-int2" else "128"
    cmd: list[str] = [
        py,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--tensor-parallel-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dist-init-addr",
        f"127.0.0.1:{dist_port}",
        "--mem-fraction-static",
        str(mem_frac),
        "--kv-cache-dtype",
        kv_dtype,
        "--prefill-attention-backend",
        prefill_be,
        "--decode-attention-backend",
        decode_be,
        "--max-running-requests",
        str(max_running_requests),
        "--max-queued-requests",
        str(max_queued_requests),
        "--page-size",
        page_size,
        "--skip-server-warmup",
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if disable_piecewise_cuda_graph:
        cmd.append("--disable-piecewise-cuda-graph")
    if piecewise_cuda_graph_max_tokens is not None:
        cmd += [
            "--piecewise-cuda-graph-max-tokens",
            str(piecewise_cuda_graph_max_tokens),
        ]
    if chunked_prefill_size is not None:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    if enable_memory_saver:
        cmd.append("--enable-memory-saver")
    if max_total_tokens is not None:
        cmd += ["--max-total-tokens", str(max_total_tokens)]
    if json_model_override_args:
        cmd += ["--json-model-override-args", json_model_override_args]
    if mode == "fp16" and not _fp16_weights_unsafe_for_model(model_path):
        cmd += ["--dtype", "float16"]
    if kv_dtype == "int2":
        cmd += ["--kv-cache-quant-group-size", str(int2_group_size)]
    if trust_remote:
        cmd.append("--trust-remote-code")
    return cmd


def _wait_health(base: str, proc: subprocess.Popen[str] | None, timeout_s: float) -> bool:
    import httpx

    deadline = time.time() + timeout_s
    probe_body = {
        "model": "default",
        "prompt": "ready",
        "max_tokens": 1,
        "temperature": 0,
    }
    with httpx.Client(timeout=5.0, trust_env=False) as client:
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            try:
                r = client.get(f"{base}/model_info")
                if r.status_code == 200:
                    info = r.json()
                    if info.get("is_generation"):
                        probe = client.post(f"{base}/v1/completions", json=probe_body, timeout=10.0)
                        if probe.status_code == 200:
                            return True
                    else:
                        return True
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2.0)
    return False


def _extract_tokens(data: dict[str, Any], fallback: int) -> int:
    usage = data.get("usage") or {}
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        if usage.get(key) is not None:
            return int(usage[key])
    if data.get("meta_info", {}).get("completion_tokens") is not None:
        return int(data["meta_info"]["completion_tokens"])
    return fallback


def _run_decode_bench(
    base: str,
    prompt: str,
    max_new_tokens: int,
    n_requests: int,
    request_api: str,
) -> tuple[float, str]:
    """Return (tok/s aggregate, error string)."""
    import httpx

    t0 = time.perf_counter()
    total_new = 0
    with httpx.Client(timeout=1800.0, trust_env=False) as client:
        for _ in range(n_requests):
            try:
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
                total_new += _extract_tokens(r.json(), max_new_tokens)
            except Exception as e:
                return 0.0, str(e)
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        return 0.0, "zero_elapsed"
    return total_new / elapsed, ""


def _decode_toks_per_sec_from_log(path: Path) -> float | None:
    """Return the best server-side decode batch throughput observed in the log."""
    metrics = parse_server_log(path)
    return metrics.decode_max_tok_s


def _metrics_from_log(
    path: Path,
    *,
    measurement_requests: int = 1,
    decode_tokens_per_request: int | None = None,
) -> ServerLogMetrics:
    return parse_server_log(
        path,
        measurement_requests=measurement_requests,
        decode_tokens_per_request=decode_tokens_per_request,
    )


def _maybe_start_profile(base: str, output_dir: Path, num_steps: int) -> None:
    if num_steps <= 0:
        return
    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "start_step": 1,
        "activities": ["CPU", "GPU"],
    }
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        client.post(f"{base}/start_profile", json=body)


def _empty_log_metrics() -> ServerLogMetrics:
    return ServerLogMetrics(
        decode_throughputs=[],
        prefill_throughputs=[],
        decode_max_tok_s=None,
        decode_first_tok_s=None,
        decode_steady_median_tok_s=None,
        decode_steady_p95_tok_s=None,
        decode_flush_median_tok_s=None,
        prefill_median_tok_s=None,
        prefill_wall_s=None,
        decode_wall_s=None,
        flush_step_fraction=None,
        effective_decode_tok_s=None,
        prefill_new_tokens=[],
        prefill_cached_tokens=[],
        cached_prefill_new_median_tokens=None,
        cached_prefill_cached_median_tokens=None,
        cached_prefill_cache_ratio_median=None,
        kv_pool_tokens=None,
        kv_k_size_gb=None,
        kv_v_size_gb=None,
        max_total_num_tokens=None,
        hp_prefix_pool_tokens=None,
        unified_mixed_kv=False,
        request_windows=[],
    )


def _mha_int_kv_bytes_per_token_estimate(
    num_kv_heads: int, head_dim: int, mode: str
) -> int:
    """Match sglang pool_configurator._mha_int_kv_bytes_per_head_pair (v_dim=head_dim)."""
    h = head_dim
    v = head_dim
    if mode in ("int8", "oscar-int8"):
        return num_kv_heads * (h + v + 2 * (h + v))
    if mode in ("int4", "oscar-int4"):
        if h % 2 != 0:
            return num_kv_heads * (h + v + 2 * (h + v))  # fallback same as int8 estimate
        return num_kv_heads * ((h + v) // 2 + 2 * (h + v))
    raise ValueError(mode)


def _dtype_size_bytes(dtype_name: str) -> int:
    if dtype_name in ("float32", "fp32"):
        return 4
    if dtype_name in ("bfloat16", "bf16", "float16", "fp16", "half"):
        return 2
    raise ValueError(f"unsupported dtype size estimate for {dtype_name!r}")


def _mode_selected_kv_gib(
    mode: str,
    geometry: KVGeometry,
    seq_len: int,
    prefix_bf16: int,
    recent_bf16: int,
    pool_tokens: int | None = None,
    max_running_requests: int = 1,
    hp_prefix_pool_tokens: int | None = None,
    group_size: int | None = 128,
    scale_dtype_bytes: int = 2,
) -> float:
    layers = geometry.layers_for_kv_estimate
    if mode == "oscar-int2":
        quant_tokens = max(seq_len, pool_tokens or 0)
        val = kv_bytes_oscar_mixed_runtime_estimate(
            layers,
            geometry.num_kv_heads,
            quant_tokens,
            geometry.head_dim,
            prefix_bf16=prefix_bf16,
            recent_bf16=recent_bf16,
            max_running_requests=max_running_requests,
            hp_prefix_pool_tokens=hp_prefix_pool_tokens,
            group_size=group_size,
            scale_dtype_bytes=scale_dtype_bytes,
        )
    elif mode == "int2":
        val = kv_bytes_int2_packed_naive(
            layers, geometry.num_kv_heads, seq_len, geometry.head_dim
        )
    elif mode in ("int8", "oscar-int8", "int4", "oscar-int4"):
        bpt = _mha_int_kv_bytes_per_token_estimate(
            geometry.num_kv_heads, geometry.head_dim, mode
        )
        val = bpt * geometry.layers_for_kv_estimate * seq_len
    elif mode in {"fp8"}:
        val = kv_bytes_bf16(layers, geometry.num_kv_heads, seq_len, geometry.head_dim) / 2
    elif mode in {"fp4"}:
        val = kv_bytes_bf16(layers, geometry.num_kv_heads, seq_len, geometry.head_dim) / 4
    else:
        val = kv_bytes_bf16(layers, geometry.num_kv_heads, seq_len, geometry.head_dim)
    return val / (1024**3)


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(BenchRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            values = []
            for name in fields:
                val = getattr(r, name)
                if isinstance(val, float):
                    values.append(f"{val:.6f}")
                else:
                    values.append("" if val is None else val)
            w.writerow(values)


def _write_md(path: Path, rows: list[BenchRow], meta: dict[str, Any]) -> None:
    lines = [
        "# OSCAR-KV-Quant bench",
        "",
        f"Generated: {meta.get('iso')}",
        f"Model path: `{meta.get('model_path')}`",
        f"Geometry: `{meta.get('geometry')}`",
        "",
        "| profile | mode | ok | kv dtype | request | prefill | request tok/s | decode steady tok/s | decode log max | prefill tok/s | baseline MiB | peak MiB | delta MiB | log |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.profile} | {r.mode} | {r.server_ok} | {r.kv_dtype_cli} | {r.request_api} | "
            f"{r.prefill_tokens} | {r.request_toks_per_sec:.2f} | "
            f"{'' if r.decode_steady_median_tok_s is None else f'{r.decode_steady_median_tok_s:.2f}'} | "
            f"{'' if r.decode_log_toks_per_sec is None else f'{r.decode_log_toks_per_sec:.2f}'} | "
            f"{'' if r.prefill_median_tok_s is None else f'{r.prefill_median_tok_s:.0f}'} | "
            f"{'' if r.baseline_mib is None else f'{r.baseline_mib:.0f}'} | "
            f"{'' if r.peak_mib_total is None else f'{r.peak_mib_total:.0f}'} | "
            f"{'' if r.peak_mib_delta is None else f'{r.peak_mib_delta:.0f}'} | "
            f"`{r.server_log_path}` |"
        )
    lines.extend(
        [
            "",
            "## End-to-end wall breakdown (last bench request window)",
            "",
            "| mode | prefill wall (s) | decode wall (s) | flush step frac | effective decode tok/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.mode} | "
            f"{'' if r.prefill_wall_s is None else f'{r.prefill_wall_s:.2f}'} | "
            f"{'' if r.decode_wall_s is None else f'{r.decode_wall_s:.2f}'} | "
            f"{'' if r.flush_step_fraction is None else f'{r.flush_step_fraction:.3f}'} | "
            f"{'' if r.effective_decode_tok_s is None else f'{r.effective_decode_tok_s:.2f}'} |"
        )
    lines.extend(
        [
            "",
            "## Decode / prefill breakdown",
            "",
            "| mode | decode first | decode steady | decode p95 | flush step | prefill median | KV pool tokens | K GB | V GB | cuda graph |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.mode} | "
            f"{'' if r.decode_first_tok_s is None else f'{r.decode_first_tok_s:.2f}'} | "
            f"{'' if r.decode_steady_median_tok_s is None else f'{r.decode_steady_median_tok_s:.2f}'} | "
            f"{'' if r.decode_steady_p95_tok_s is None else f'{r.decode_steady_p95_tok_s:.2f}'} | "
            f"{'' if r.decode_flush_median_tok_s is None else f'{r.decode_flush_median_tok_s:.2f}'} | "
            f"{'' if r.prefill_median_tok_s is None else f'{r.prefill_median_tok_s:.0f}'} | "
            f"{'' if r.kv_pool_tokens is None else r.kv_pool_tokens} | "
            f"{'' if r.kv_k_size_gb is None else f'{r.kv_k_size_gb:.3f}'} | "
            f"{'' if r.kv_v_size_gb is None else f'{r.kv_v_size_gb:.3f}'} | "
            f"{r.cuda_graph_enabled} |"
        )
    lines.extend(
        [
            "",
            "## KV estimate (K+V runtime layout)",
            "",
            "| mode | selected KV estimate (GiB) | bf16 KV (GiB) | ratio | note |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        ratio = (
            r.kv_theory_selected_gib / r.kv_theory_bf16_gib
            if r.kv_theory_bf16_gib > 0
            else 0.0
        )
        lines.append(
            f"| {r.mode} | {r.kv_theory_selected_gib:.4f} | "
            f"{r.kv_theory_bf16_gib:.4f} | {ratio:.3f} | {r.kv_mode_note} |"
        )
    failures = [r for r in rows if not r.server_ok or r.error]
    if failures:
        lines.extend(["", "## Failures", ""])
        for r in failures:
            lines.append(f"### {r.mode}")
            lines.append("")
            lines.append(f"Error: `{r.error}`")
            if r.log_tail:
                lines.append("")
                lines.append("```text")
                lines.append(r.log_tail[-4000:])
                lines.append("```")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `peak_mib_delta` is total GPU memory delta from the pre-run nvidia-smi baseline, not an exact per-process allocation.",
            "- `request_toks_per_sec` divides generated tokens by whole HTTP request time, so long-prefill runs include prefill cost.",
            "- `prefill_wall_s` / `decode_wall_s` come from server log timestamps on the last `--bench-requests` window.",
            "- `flush_step_fraction` is the share of decode log samples (after the first) classified as flush outliers.",
            "- `effective_decode_tok_s` is `--max-new-tokens` divided by `decode_wall_s` (includes flush steps).",
            "- `decode_log_toks_per_sec` is the max server-side decode batch throughput from the log.",
            "- `decode_steady_median_tok_s` excludes the first decode step and low-throughput flush steps.",
            "- KV estimates include the known runtime KV layout for int2/oscar-int2, but exclude model weights, non-KV allocator reserve, attention workspaces, and kernel caches.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_modes(arg: str) -> Iterator[str]:
    for m in arg.split(","):
        m = m.strip()
        if m:
            yield m


def _row_for_skipped(
    args: argparse.Namespace,
    mode: str,
    kv: str,
    error: str,
    geometry: KVGeometry,
    seq: int,
    theory_bf16: float,
    selected: float,
) -> BenchRow:
    return BenchRow(
        profile=args.profile,
        mode=mode,
        kv_dtype_cli=kv,
        kv_mode_note=MODE_NOTES.get(mode, ""),
        request_api=args.request_api,
        prefill_tokens=args.prefill_tokens or PRESET_TOKENS[args.preset],
        max_new_tokens=args.max_new_tokens,
        decode_toks_per_sec=0.0,
        request_toks_per_sec=0.0,
        decode_log_toks_per_sec=None,
        baseline_mib=None,
        peak_mib_total=None,
        peak_mib_delta=None,
        server_pid=None,
        server_log_path="",
        server_ok=False,
        error=error,
        log_tail="",
        num_layers=geometry.num_layers,
        num_attention_layers=geometry.num_attention_layers,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        kv_theory_bf16_gib=theory_bf16,
        kv_theory_selected_gib=selected,
        decode_first_tok_s=None,
        decode_steady_median_tok_s=None,
        decode_steady_p95_tok_s=None,
        decode_flush_median_tok_s=None,
        prefill_median_tok_s=None,
        prefill_wall_s=None,
        decode_wall_s=None,
        flush_step_fraction=None,
        effective_decode_tok_s=None,
        kv_pool_tokens=None,
        kv_k_size_gb=None,
        kv_v_size_gb=None,
        cuda_graph_enabled=_cuda_graph_enabled(args, mode),
        profile_dir="",
    )


def _make_bench_row(
    *,
    args: argparse.Namespace,
    mode: str,
    kv: str,
    prefill_n: int,
    theory_bf16: float,
    selected: float,
    geometry: KVGeometry,
    log_path: Path,
    metrics: ServerLogMetrics,
    tok_s: float,
    baseline: float | None,
    peak: float | None,
    delta: float | None,
    proc: subprocess.Popen[str] | None,
    ok: bool,
    err: str,
    log_tail: str,
) -> BenchRow:
    profile_dir = ""
    if args.profile_steps > 0:
        profile_dir = str(args.results_dir / f"profile_{args.profile}_{mode}")
    return BenchRow(
        profile=args.profile,
        mode=mode,
        kv_dtype_cli=kv,
        kv_mode_note=MODE_NOTES.get(mode, ""),
        request_api=args.request_api,
        prefill_tokens=prefill_n,
        max_new_tokens=args.max_new_tokens,
        decode_toks_per_sec=tok_s,
        request_toks_per_sec=tok_s,
        decode_log_toks_per_sec=metrics.decode_max_tok_s,
        baseline_mib=baseline,
        peak_mib_total=peak,
        peak_mib_delta=delta,
        server_pid=proc.pid if proc else None,
        server_log_path=str(log_path),
        server_ok=ok and not err,
        error=err,
        log_tail=log_tail if err else "",
        num_layers=geometry.num_layers,
        num_attention_layers=geometry.num_attention_layers,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        kv_theory_bf16_gib=theory_bf16,
        kv_theory_selected_gib=selected,
        decode_first_tok_s=metrics.decode_first_tok_s,
        decode_steady_median_tok_s=metrics.decode_steady_median_tok_s,
        decode_steady_p95_tok_s=metrics.decode_steady_p95_tok_s,
        decode_flush_median_tok_s=metrics.decode_flush_median_tok_s,
        prefill_median_tok_s=metrics.prefill_median_tok_s,
        prefill_wall_s=metrics.prefill_wall_s,
        decode_wall_s=metrics.decode_wall_s,
        flush_step_fraction=metrics.flush_step_fraction,
        effective_decode_tok_s=metrics.effective_decode_tok_s,
        kv_pool_tokens=metrics.kv_pool_tokens,
        kv_k_size_gb=metrics.kv_k_size_gb,
        kv_v_size_gb=metrics.kv_v_size_gb,
        cuda_graph_enabled=_cuda_graph_enabled(args, mode),
        profile_dir=profile_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="granite")
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--preset", choices=sorted(PRESET_TOKENS.keys()), default="short")
    ap.add_argument("--prefill-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--modes",
        default="bf16,fp8,int2",
        help="Comma list: bf16, fp8, fp4, int8, int4, int2, oscar-int2 "
        "(oscar-int8 / oscar-int4 use SGLANG_OSCAR_ROTATE_QUANT_KV + Triton prefill/decode)",
    )
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
    ap.add_argument("--port", type=int, default=31888)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Forwarded to SGLang to cap the total KV pool tokens for fair memory comparisons.",
    )
    ap.add_argument("--warmup-requests", type=int, default=1)
    ap.add_argument("--bench-requests", type=int, default=2)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--request-api", choices=["chat", "completions", "generate"], default="completions")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=None)
    ap.add_argument(
        "--prefix-bf16-tokens",
        type=int,
        default=64,
        help="OSCAR mixed-KV high-precision prefix tokens; also forwarded to runtime.",
    )
    ap.add_argument(
        "--recent-bf16-tokens",
        default="auto",
        help=(
            "OSCAR mixed-KV high-precision recent tokens; integer or 'auto'. "
            "auto uses 64 for <=64 generated tokens, otherwise 256."
        ),
    )
    ap.add_argument(
        "--hp-prefix-pool-tokens",
        type=int,
        default=None,
        help=(
            "Forward SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS for oscar-int2. "
            "Use a small value such as 64/256 for single-request memory comparisons."
        ),
    )
    ap.add_argument(
        "--mixed-kv-max-quant-tokens",
        type=int,
        default=None,
        help=(
            "Forward SGLANG_MIXED_KV_MAX_QUANT_TOKENS for oscar-int2. "
            "Defaults to the effective --max-total-tokens/fair single-request cap; "
            "set 0 to let mixed int2 auto-size to available memory."
        ),
    )
    ap.add_argument(
        "--max-running-requests",
        type=int,
        default=1,
        help=(
            "Forwarded to SGLang --max-running-requests. Defaults to 1 so "
            "oscar-int2 does not reserve extra HP-recent slabs during memory "
            "comparisons."
        ),
    )
    ap.add_argument(
        "--max-queued-requests",
        type=int,
        default=8,
        help="Forwarded to SGLang --max-queued-requests.",
    )
    ap.add_argument("--prefill-attention-backend", default=None)
    ap.add_argument("--decode-attention-backend", default=None)
    ap.add_argument(
        "--json-model-override-args",
        default=None,
        help="JSON forwarded to SGLang --json-model-override-args.",
    )
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help=(
            "Allow SGLang CUDA graph capture. Enabled by default for oscar-* modes; "
            "other modes keep graphs disabled unless this flag is set."
        ),
    )
    ap.add_argument(
        "--disable-oscar-cuda-graph",
        action="store_true",
        help="Disable the oscar-* CUDA graph default for memory-priority experiments.",
    )
    ap.add_argument(
        "--enable-piecewise-cuda-graph",
        action="store_true",
        help="Allow SGLang piecewise CUDA graph capture (requires --enable-cuda-graph).",
    )
    ap.add_argument(
        "--piecewise-cuda-graph-max-tokens",
        type=int,
        default=None,
        help=(
            "Cap piecewise CUDA graph capture size (VRAM safety on 8GB GPUs). "
            "Try 1024 when 2048 shapes OOM."
        ),
    )
    ap.add_argument(
        "--enable-memory-saver",
        action="store_true",
        help="Forward SGLang --enable-memory-saver for memory-priority experiments.",
    )
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="If >0, call /start_profile for this many decode steps during benchmark requests.",
    )
    ap.add_argument(
        "--mixed-kv-hp-max-splits",
        type=int,
        default=4,
        help="Forward SGLANG_MIXED_KV_HP_MAX_SPLITS for oscar-int2.",
    )
    ap.add_argument(
        "--mixed-kv-scale-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Forward SGLANG_MIXED_KV_SCALE_DTYPE for oscar-int2.",
    )
    ap.add_argument(
        "--enable-fused-rotate-clip-quant",
        action="store_true",
        help="Set SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT=1 for oscar-int2 extend writes.",
    )
    ap.add_argument(
        "--oscar-min-prefill-tokens",
        type=int,
        default=0,
        help=(
            "If >0, skip or fall back from oscar-int2 when prefill tokens are "
            "below this threshold; useful because short contexts do not amortize "
            "mixed-KV fixed costs."
        ),
    )
    ap.add_argument(
        "--oscar-short-context-fallback",
        choices=["skip", "int2", "bf16"],
        default="skip",
        help="Behavior when --oscar-min-prefill-tokens skips oscar-int2.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    args = ap.parse_args()
    if not getattr(args, "enable_fused_rotate_clip_quant", False):
        args.enable_fused_rotate_clip_quant = True

    profile = PROFILES[args.profile]
    model_path = Path(args.model_path or profile.default_model_path)
    local_ok = model_path.is_dir() and (model_path / "config.json").is_file()
    if not local_ok:
        from transformers import AutoConfig

        try:
            AutoConfig.from_pretrained(
                str(model_path), trust_remote_code=True, local_files_only=False
            )
        except Exception as e:
            print(f"Cannot load model config from {model_path}: {e}", flush=True)
            raise SystemExit(2)

    geometry = resolve_kv_geometry(model_path, profile)
    if args.num_layers:
        geometry.num_layers = args.num_layers
        geometry.num_attention_layers = args.num_layers
    if args.num_kv_heads:
        geometry.num_kv_heads = args.num_kv_heads
    if args.head_dim:
        geometry.head_dim = args.head_dim
    int2_group_size = _int2_group_size_for_head_dim(geometry.head_dim)

    prefill_n = args.prefill_tokens or PRESET_TOKENS[args.preset]
    seq = prefill_n + args.max_new_tokens
    recent_bf16_tokens = _resolve_recent_bf16_tokens(
        args.recent_bf16_tokens, max_new_tokens=args.max_new_tokens
    )
    layers_for_est = geometry.layers_for_kv_estimate
    theory_bf16 = kv_bytes_bf16(
        layers_for_est, geometry.num_kv_heads, seq, geometry.head_dim
    ) / (1024**3)

    print(
        f"[bench] geometry layers={geometry.num_layers} "
        f"attention_layers={geometry.num_attention_layers} "
        f"kv_heads={geometry.num_kv_heads} head_dim={geometry.head_dim} "
        f"seq~={seq} => KV_bf16≈{fmt_gib(theory_bf16 * (1024**3))} GiB",
        flush=True,
    )

    py = args.python
    prompt = "" if args.dry_run else _build_prefill_text(model_path, prefill_n)
    base = f"http://127.0.0.1:{args.port}"
    rows: list[BenchRow] = []

    for mode in iter_modes(args.modes):
        fallback_mode = _short_context_oscar_fallback_mode(
            mode,
            prefill_n,
            args.oscar_min_prefill_tokens,
            args.oscar_short_context_fallback,
        )
        if fallback_mode == "skip":
            kv = MODE_KV[mode]
            selected = _mode_selected_kv_gib(
                mode,
                geometry,
                seq,
                args.prefix_bf16_tokens,
                recent_bf16_tokens,
                max_running_requests=args.max_running_requests,
                hp_prefix_pool_tokens=args.hp_prefix_pool_tokens,
                group_size=int2_group_size,
                scale_dtype_bytes=_dtype_size_bytes(args.mixed_kv_scale_dtype),
            )
            msg = (
                f"oscar-int2 skipped for short context: prefill_tokens={prefill_n} "
                f"< oscar_min_prefill_tokens={args.oscar_min_prefill_tokens}"
            )
            rows.append(
                _row_for_skipped(args, mode, kv, msg, geometry, seq, theory_bf16, selected)
            )
            print(f"[bench] skip {mode}: {msg}", flush=True)
            continue
        if fallback_mode in ("int2", "bf16"):
            print(
                f"[bench] fallback oscar-int2 -> {fallback_mode}: "
                f"prefill_tokens={prefill_n} < "
                f"oscar_min_prefill_tokens={args.oscar_min_prefill_tokens}",
                flush=True,
            )
            mode = fallback_mode

        kv = MODE_KV.get(mode)
        if kv is None:
            print(f"[bench] skip unknown mode {mode}", flush=True)
            continue

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
        if mode == "oscar-int2":
            mixed_kv_max_quant_tokens = (
                args.mixed_kv_max_quant_tokens
                if args.mixed_kv_max_quant_tokens is not None
                else max_total_tokens
            )

        selected = _mode_selected_kv_gib(
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

        env = os.environ.copy()
        if mode == "oscar-int2":
            valid, msg = _validate_rotation_files(
                args.rot_dir,
                mode,
                k_rotation_filename=args.k_rotation_filename,
                v_rotation_filename=args.v_rotation_filename,
            )
            if not valid:
                rows.append(
                    _row_for_skipped(args, mode, kv, msg, geometry, seq, theory_bf16, selected)
                )
                print(f"[bench] skip {mode}: {msg}", flush=True)
                continue
            assert args.rot_dir is not None
            env.update(
                _oscar_env(
                    args.rot_dir,
                    args.prefix_bf16_tokens,
                    recent_bf16_tokens,
                    args.hp_prefix_pool_tokens,
                    max_quant_tokens=mixed_kv_max_quant_tokens,
                    hp_max_splits=args.mixed_kv_hp_max_splits,
                    scale_dtype=args.mixed_kv_scale_dtype,
                    fused_rotate_clip_quant=args.enable_fused_rotate_clip_quant,
                    k_rotation_filename=args.k_rotation_filename,
                    v_rotation_filename=args.v_rotation_filename,
                    k_clip_ratio=args.oscar_k_clip_ratio,
                    v_clip_ratio=args.oscar_v_clip_ratio,
                    lloyd_max=args.lloyd_max,
                )
            )
        elif mode in ("oscar-int8", "oscar-int4"):
            valid, msg = _validate_rotation_files(
                args.rot_dir,
                mode,
                k_rotation_filename=args.k_rotation_filename,
                v_rotation_filename=args.v_rotation_filename,
            )
            if not valid:
                rows.append(
                    _row_for_skipped(args, mode, kv, msg, geometry, seq, theory_bf16, selected)
                )
                print(f"[bench] skip {mode}: {msg}", flush=True)
                continue
            assert args.rot_dir is not None
            env.update(
                _oscar_quant_kv_env(
                    args.rot_dir,
                    k_rotation_filename=args.k_rotation_filename,
                    v_rotation_filename=args.v_rotation_filename,
                )
            )

        cmd = _server_cmd(
            py,
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
            json_model_override_args=args.json_model_override_args,
            max_running_requests=args.max_running_requests,
            max_queued_requests=args.max_queued_requests,
            disable_cuda_graph=not _cuda_graph_enabled(args, mode),
            disable_piecewise_cuda_graph=not args.enable_piecewise_cuda_graph,
            piecewise_cuda_graph_max_tokens=args.piecewise_cuda_graph_max_tokens,
            enable_memory_saver=args.enable_memory_saver,
            int2_group_size=int2_group_size,
        )

        log_path = args.results_dir / f"server_{args.profile}_{mode}.log"
        if args.dry_run:
            print(f"[bench:dry-run] mode={mode} cmd={' '.join(cmd)}", flush=True)
            if mode.startswith("oscar-"):
                relevant = {k: env[k] for k in sorted(env) if k.startswith("SGLANG_")}
                print("[bench:dry-run] OSCAR env=" + json.dumps(relevant, indent=2), flush=True)
            rows.append(
                _make_bench_row(
                    args=args,
                    mode=mode,
                    kv=kv,
                    prefill_n=prefill_n,
                    theory_bf16=theory_bf16,
                    selected=selected,
                    geometry=geometry,
                    log_path=log_path,
                    metrics=_empty_log_metrics(),
                    tok_s=0.0,
                    baseline=None,
                    peak=None,
                    delta=None,
                    proc=None,
                    ok=True,
                    err="dry-run",
                    log_tail="",
                )
            )
            continue

        print(f"[bench] start server mode={mode} kv={kv}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        baseline = _gpu_memory_used_mib()
        samples: list[float] = []
        stop_poll = threading.Event()
        poller = threading.Thread(target=_poll_nvidia_smi_mib, args=(samples, stop_poll))
        poller.start()
        proc: subprocess.Popen[str] | None = None
        err = ""
        tok_s = 0.0
        ok = False
        try:
            with log_path.open("w") as logf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
            print(f"[bench] wait health mode={mode}", flush=True)
            if not _wait_health(base, proc, args.health_timeout):
                err = "server_health_timeout"
            else:
                ok = True
                print(f"[bench] warmup mode={mode}", flush=True)
                _, werr = _run_decode_bench(
                    base, prompt, args.max_new_tokens, args.warmup_requests, args.request_api
                )
                if werr:
                    err = f"warmup:{werr}"
                else:
                    if args.profile_steps > 0:
                        profile_out = args.results_dir / f"profile_{args.profile}_{mode}"
                        _maybe_start_profile(base, profile_out, args.profile_steps)
                    print(f"[bench] measure mode={mode}", flush=True)
                    tok_s, berr = _run_decode_bench(
                        base,
                        prompt,
                        args.max_new_tokens,
                        args.bench_requests,
                        args.request_api,
                    )
                    if berr:
                        err = berr
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
        peak = max(samples) if samples else None
        delta = None if peak is None or baseline is None else max(0.0, peak - baseline)
        log_tail = _read_tail(log_path)
        metrics = _metrics_from_log(
            log_path,
            measurement_requests=args.bench_requests,
            decode_tokens_per_request=args.max_new_tokens,
        )
        decode_log_tok_s = metrics.decode_max_tok_s
        if err == "server_health_timeout" and log_tail:
            err = f"{err}: {log_tail.splitlines()[-1][:240]}"
        rows.append(
            _make_bench_row(
                args=args,
                mode=mode,
                kv=kv,
                prefill_n=prefill_n,
                theory_bf16=theory_bf16,
                selected=selected,
                geometry=geometry,
                log_path=log_path,
                metrics=metrics,
                tok_s=tok_s,
                baseline=baseline,
                peak=peak,
                delta=delta,
                proc=proc,
                ok=ok,
                err=err,
                log_tail=log_tail if err else "",
            )
        )
        print(
            f"[bench] done mode={mode} request_tok/s={tok_s:.2f} "
            f"decode_steady_tok/s={metrics.decode_steady_median_tok_s} "
            f"decode_log_max_tok/s={decode_log_tok_s} peak_mib={peak} "
            f"delta_mib={delta} err={err!r}",
            flush=True,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"bench_{args.profile}_{ts}.csv"
    md_path = args.results_dir / f"bench_{args.profile}_{ts}.md"
    _write_csv(csv_path, rows)
    meta = {
        "iso": ts,
        "model_path": to_repo_relative(model_path),
        "geometry": asdict(geometry),
        "raw_config_keys": list(load_model_config(model_path).keys())[:12],
    }
    _write_md(md_path, rows, meta)
    print(f"[bench] wrote {csv_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
