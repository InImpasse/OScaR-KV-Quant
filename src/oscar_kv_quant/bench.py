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
    kv_bytes_oscar_mixed_estimate,
)
from oscar_kv_quant.profiles import PROFILES, KVGeometry, load_model_config, resolve_kv_geometry

PRESET_TOKENS = {"short": 512, "medium": 2048, "long": 8192}

# Logical names -> sglang --kv-cache-dtype (see docs/MODEL_SUPPORT.md)
MODE_KV = {
    "bf16": "bf16",
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
    "fp8": "8-bit floating KV cache (fp8_e4m3)",
    "int8": "Symmetric int8 KV + internal bf16 shadow for Triton (true integer storage)",
    "fp4": "4-bit floating KV cache (fp4_e2m1 / MXFP4), hardware dependent",
    "int4": "Symmetric int4 (nibble-packed) KV + bf16 shadow for Triton; head_dim must be even",
    "int2": "SGLang Triton INT2 KV cache without OSCAR rotations",
    "oscar-int2": "OSCAR INT2 mixed KV windows with rotation files",
    "oscar-int8": "int8 KV + Oscar (SGLANG_OSCAR_ROTATE_QUANT_KV + rotation checkpoints)",
    "oscar-int4": "int4 KV + Oscar (SGLANG_OSCAR_ROTATE_QUANT_KV + rotation checkpoints)",
}


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


def _rotation_paths(rot_dir: Path) -> tuple[Path, Path]:
    return (
        rot_dir / os.environ.get("K_ROT_FILENAME", "k_rotation_qqt_r_h_pbr.pt"),
        rot_dir / os.environ.get("V_ROT_FILENAME", "v_rotation_sst_r_h_pbr.pt"),
    )


def _validate_rotation_files(
    rot_dir: Path | None, mode_label: str = "oscar"
) -> tuple[bool, str]:
    if rot_dir is None:
        return False, f"{mode_label} requires --rot-dir"
    k, v = _rotation_paths(rot_dir)
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


def _default_prefill_backend(mode: str) -> str:
    if mode in ("int2", "oscar-int2"):
        # SM120 consumer Blackwell cannot run the current FA3/FA4 paths used by
        # this INT2 dense-prefill code. Triton stays on the quantized KV path
        # and falls back to SDPA only for the incompatible dense prefill step.
        return "triton" if _is_cuda_sm120() else "fa3"
    return "triton"


def _oscar_quant_kv_env(rot_dir: Path) -> dict[str, str]:
    """Env for FP8/FP4 KV + learned Oscar rotations (plain MHA pool, not mixed int2)."""
    k, v = _rotation_paths(rot_dir)
    return {
        "SGLANG_OSCAR_ROTATE_QUANT_KV": "1",
        "SGLANG_OSCAR_ABSORB_V_ROTATION": "1",
        "SGLANG_OSCAR_K_ROTATION_PATH": str(k),
        "SGLANG_OSCAR_V_ROTATION_PATH": str(v),
    }


def _oscar_env(rot_dir: Path) -> dict[str, str]:
    k, v = _rotation_paths(rot_dir)
    return {
        "SGLANG_ENABLE_MIXED_KV_WINDOWS": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_OSCAR_ABSORB_V_ROTATION": "1",
        "SGLANG_MIXED_KV_HP_MAX_SPLITS": "8",
        "SGLANG_MIXED_KV_PREFIX_TOKENS": os.environ.get(
            "SGLANG_MIXED_KV_PREFIX_TOKENS", "64"
        ),
        "SGLANG_MIXED_KV_RECENT_TOKENS": os.environ.get(
            "SGLANG_MIXED_KV_RECENT_TOKENS", "256"
        ),
        "SGLANG_MIXED_KV_HP_DTYPE": "bfloat16",
        "SGLANG_MIXED_KV_SCALE_DTYPE": "float32",
        "SGLANG_OSCAR_K_ROTATION_PATH": str(k),
        "SGLANG_OSCAR_V_ROTATION_PATH": str(v),
        "SGLANG_OSCAR_K_CLIP_RATIO": os.environ.get("SGLANG_OSCAR_K_CLIP_RATIO", "0.96"),
        "SGLANG_OSCAR_V_CLIP_RATIO": os.environ.get("SGLANG_OSCAR_V_CLIP_RATIO", "0.92"),
        "SGLANG_LLOYD_MAX": os.environ.get("SGLANG_LLOYD_MAX", "0"),
    }


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
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--max-running-requests",
        "4",
        "--max-queued-requests",
        "8",
        "--page-size",
        page_size,
    ]
    if max_total_tokens is not None:
        cmd += ["--max-total-tokens", str(max_total_tokens)]
    if kv_dtype == "int2":
        cmd += ["--kv-cache-quant-group-size", "128"]
    if trust_remote:
        cmd.append("--trust-remote-code")
    return cmd


def _wait_health(base: str, proc: subprocess.Popen[str] | None, timeout_s: float) -> bool:
    import httpx

    deadline = time.time() + timeout_s
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                return False
            try:
                r = client.get(f"{base}/health")
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
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
    with httpx.Client(timeout=300.0) as client:
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


def _mode_selected_kv_gib(
    mode: str,
    geometry: KVGeometry,
    seq_len: int,
    prefix_bf16: int,
    recent_bf16: int,
) -> float:
    layers = geometry.layers_for_kv_estimate
    if mode == "oscar-int2":
        val = kv_bytes_oscar_mixed_estimate(
            layers,
            geometry.num_kv_heads,
            seq_len,
            geometry.head_dim,
            prefix_bf16=prefix_bf16,
            recent_bf16=recent_bf16,
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
        "| profile | mode | ok | kv dtype | request | prefill | tok/s | baseline MiB | peak MiB | delta MiB | log |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.profile} | {r.mode} | {r.server_ok} | {r.kv_dtype_cli} | {r.request_api} | "
            f"{r.prefill_tokens} | {r.decode_toks_per_sec:.2f} | "
            f"{'' if r.baseline_mib is None else f'{r.baseline_mib:.0f}'} | "
            f"{'' if r.peak_mib_total is None else f'{r.peak_mib_total:.0f}'} | "
            f"{'' if r.peak_mib_delta is None else f'{r.peak_mib_delta:.0f}'} | "
            f"`{r.server_log_path}` |"
        )
    lines.extend(
        [
            "",
            "## KV theory (K+V only)",
            "",
            "| mode | selected KV estimate (GiB) | bf16 KV (GiB) | note |",
            "|---|---:|---:|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.mode} | {r.kv_theory_selected_gib:.4f} | {r.kv_theory_bf16_gib:.4f} | {r.kv_mode_note} |"
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
            "- Theoretical KV estimates exclude model weights, allocator reserve, attention workspaces, and kernel caches.",
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
    ap.add_argument("--prefix-bf16-tokens", type=int, default=64)
    ap.add_argument("--recent-bf16-tokens", type=int, default=256)
    ap.add_argument("--prefill-attention-backend", default=None)
    ap.add_argument("--decode-attention-backend", default=None)
    ap.add_argument("--health-timeout", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    args = ap.parse_args()

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

    prefill_n = args.prefill_tokens or PRESET_TOKENS[args.preset]
    seq = prefill_n + args.max_new_tokens
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

    py = os.environ.get("OSCAR_KV_PYTHON", sys.executable)
    prompt = "" if args.dry_run else _build_prefill_text(model_path, prefill_n)
    base = f"http://127.0.0.1:{args.port}"
    rows: list[BenchRow] = []

    for mode in iter_modes(args.modes):
        kv = MODE_KV.get(mode)
        if kv is None:
            print(f"[bench] skip unknown mode {mode}", flush=True)
            continue

        selected = _mode_selected_kv_gib(
            mode,
            geometry,
            seq,
            args.prefix_bf16_tokens,
            args.recent_bf16_tokens,
        )

        env = os.environ.copy()
        if mode == "oscar-int2":
            valid, msg = _validate_rotation_files(args.rot_dir, mode)
            if not valid:
                rows.append(
                    _row_for_skipped(args, mode, kv, msg, geometry, seq, theory_bf16, selected)
                )
                print(f"[bench] skip {mode}: {msg}", flush=True)
                continue
            assert args.rot_dir is not None
            env.update(_oscar_env(args.rot_dir))
        elif mode in ("oscar-int8", "oscar-int4"):
            valid, msg = _validate_rotation_files(args.rot_dir, mode)
            if not valid:
                rows.append(
                    _row_for_skipped(args, mode, kv, msg, geometry, seq, theory_bf16, selected)
                )
                print(f"[bench] skip {mode}: {msg}", flush=True)
                continue
            assert args.rot_dir is not None
            env.update(_oscar_quant_kv_env(args.rot_dir))

        cmd = _server_cmd(
            py,
            model_path,
            args.port,
            args.port + 1000,
            kv,
            mode,
            args.mem_fraction_static,
            args.max_total_tokens,
            trust_remote=not args.no_trust_remote_code,
            prefill_backend=args.prefill_attention_backend,
            decode_backend=args.decode_attention_backend,
        )

        log_path = args.results_dir / f"server_{args.profile}_{mode}.log"
        if args.dry_run:
            print(f"[bench:dry-run] mode={mode} cmd={' '.join(cmd)}", flush=True)
            if mode.startswith("oscar-"):
                relevant = {k: env[k] for k in sorted(env) if k.startswith("SGLANG_")}
                print("[bench:dry-run] OSCAR env=" + json.dumps(relevant, indent=2), flush=True)
            rows.append(
                BenchRow(
                    profile=args.profile,
                    mode=mode,
                    kv_dtype_cli=kv,
                    kv_mode_note=MODE_NOTES.get(mode, ""),
                    request_api=args.request_api,
                    prefill_tokens=prefill_n,
                    max_new_tokens=args.max_new_tokens,
                    decode_toks_per_sec=0.0,
                    baseline_mib=None,
                    peak_mib_total=None,
                    peak_mib_delta=None,
                    server_pid=None,
                    server_log_path=str(log_path),
                    server_ok=True,
                    error="dry-run",
                    log_tail="",
                    num_layers=geometry.num_layers,
                    num_attention_layers=geometry.num_attention_layers,
                    num_kv_heads=geometry.num_kv_heads,
                    head_dim=geometry.head_dim,
                    kv_theory_bf16_gib=theory_bf16,
                    kv_theory_selected_gib=selected,
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
            if not _wait_health(base, proc, args.health_timeout):
                err = "server_health_timeout"
            else:
                ok = True
                _, werr = _run_decode_bench(
                    base, prompt, args.max_new_tokens, args.warmup_requests, args.request_api
                )
                if werr:
                    err = f"warmup:{werr}"
                else:
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
        if err == "server_health_timeout" and log_tail:
            err = f"{err}: {log_tail.splitlines()[-1][:240]}"
        rows.append(
            BenchRow(
                profile=args.profile,
                mode=mode,
                kv_dtype_cli=kv,
                kv_mode_note=MODE_NOTES.get(mode, ""),
                request_api=args.request_api,
                prefill_tokens=prefill_n,
                max_new_tokens=args.max_new_tokens,
                decode_toks_per_sec=tok_s,
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
            )
        )
        print(
            f"[bench] done mode={mode} tok/s={tok_s:.2f} peak_mib={peak} delta_mib={delta} err={err!r}",
            flush=True,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"bench_{args.profile}_{ts}.csv"
    md_path = args.results_dir / f"bench_{args.profile}_{ts}.md"
    _write_csv(csv_path, rows)
    meta = {
        "iso": ts,
        "model_path": str(model_path),
        "geometry": asdict(geometry),
        "raw_config_keys": list(load_model_config(model_path).keys())[:12],
    }
    _write_md(md_path, rows, meta)
    print(f"[bench] wrote {csv_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
