"""Environment and SGLang server probes for OSCAR-KV-Quant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

from oscar_kv_quant.paths import repo_cache_dir


@dataclass
class ProbeStatus:
    env_ok: bool = False
    sglang_import_ok: bool = False
    flashinfer_import_ok: bool = False
    dummy_server_ok: bool | None = None
    model_server_ok: bool | None = None
    int2_server_ok: bool | None = None


def _tail(path: Path, n: int = 50) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def _health_ok(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10)
        return True
    except Exception:
        return False


def _served_model_name(model_path: str) -> str:
    name = Path(model_path).name or model_path
    return name.replace(":", "_")


def _run_server_probe(
    *,
    model_path: str,
    port: int,
    kv_cache_dtype: str,
    prefill_attention_backend: str,
    decode_attention_backend: str,
    timeout_s: int,
    mem_fraction_static: float,
    extra_args: list[str] | None = None,
) -> tuple[bool, Path]:
    log_path = repo_cache_dir() / (
        f"oscar_kv_probe_{Path(model_path).name}_{kv_cache_dtype}_{port}.log"
    )
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        _served_model_name(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dist-init-addr",
        f"127.0.0.1:{port + 1000}",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        "2048",
        "--page-size",
        "1",
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--prefill-attention-backend",
        prefill_attention_backend,
        "--decode-attention-backend",
        decode_attention_backend,
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--trust-remote-code",
    ]
    if kv_cache_dtype == "int2":
        cmd += ["--kv-cache-quant-group-size", "128"]
    if extra_args:
        cmd += extra_args
    print("== server probe cmd ==", " ".join(cmd), flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy())
    ok = False
    try:
        for _ in range(timeout_s):
            if _health_ok(port):
                ok = True
                break
            if proc.poll() is not None:
                break
            time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("== server probe /health ==", "OK" if ok else "FAIL")
        print("== server probe log ==", log_path)
        tail = _tail(log_path)
        if tail:
            print("== server probe log tail ==\n" + tail)
    return ok, log_path


def main() -> None:
    p = argparse.ArgumentParser(description="Probe CUDA + SGLang for OSCAR-KV-Quant")
    p.add_argument("--try-dummy-server", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--try-model-server", action="store_true")
    p.add_argument("--try-int2", action="store_true")
    p.add_argument("--kv-cache-dtype", default="bf16")
    p.add_argument("--prefill-attention-backend", default="triton")
    p.add_argument("--decode-attention-backend", default="triton")
    p.add_argument("--port", type=int, default=31789)
    p.add_argument("--timeout-s", type=int, default=90)
    p.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = p.parse_args()

    status = ProbeStatus()
    print("== Python ==", sys.version)
    print("== Executable ==", sys.executable)

    try:
        import torch

        print("== torch ==", torch.__version__)
        print("== torch.cuda.is_available ==", torch.cuda.is_available())
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            print(f"== GPU == {name} sm_{major}{minor}")
            print(f"== VRAM total == {props.total_memory / (1024**3):.2f} GiB")
            status.env_ok = True
        else:
            status.env_ok = False
    except Exception as e:
        print("== torch ERROR ==", e)

    try:
        import sglang

        print("== sglang ==", getattr(sglang, "__version__", "unknown"))
        print("== sglang file ==", sglang.__file__)
        status.sglang_import_ok = True
    except Exception as e:
        print("== sglang import FAILED ==", e)
        print("  Hint: ./scripts/install_sglang_os.sh from repo root")

    try:
        import flashinfer  # noqa: F401

        print("== flashinfer == OK")
        status.flashinfer_import_ok = True
    except Exception as e:
        print("== flashinfer ==", e)

    if args.try_dummy_server:
        ok, _ = _run_server_probe(
            model_path="dummy",
            port=args.port,
            kv_cache_dtype="auto",
            prefill_attention_backend=args.prefill_attention_backend,
            decode_attention_backend=args.decode_attention_backend,
            timeout_s=args.timeout_s,
            mem_fraction_static=0.3,
        )
        status.dummy_server_ok = ok

    if args.try_model_server:
        if not args.model_path:
            print("== model server SKIP == --model-path is required")
            status.model_server_ok = False
        else:
            ok, _ = _run_server_probe(
                model_path=args.model_path,
                port=args.port,
                kv_cache_dtype=args.kv_cache_dtype,
                prefill_attention_backend=args.prefill_attention_backend,
                decode_attention_backend=args.decode_attention_backend,
                timeout_s=args.timeout_s,
                mem_fraction_static=args.mem_fraction_static,
            )
            status.model_server_ok = ok

    if args.try_int2:
        if not args.model_path:
            print("== int2 server SKIP == --model-path is required")
            status.int2_server_ok = False
        else:
            ok, _ = _run_server_probe(
                model_path=args.model_path,
                port=args.port + 10,
                kv_cache_dtype="int2",
                prefill_attention_backend=args.prefill_attention_backend,
                decode_attention_backend="triton",
                timeout_s=args.timeout_s,
                mem_fraction_static=args.mem_fraction_static,
            )
            status.int2_server_ok = ok

    print("== probe status ==")
    print(json.dumps(asdict(status), indent=2))


if __name__ == "__main__":
    main()
