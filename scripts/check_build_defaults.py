#!/usr/bin/env python3
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    Path("scripts/build_llamacpp.sh"): [
        'BUILD_DIR="${BUILD_DIR:-$OSCAR_DIR/build-cuda}"',
        'LLAMACPP_CMAKE_ARGS="${LLAMACPP_CMAKE_ARGS:--DLLAMA_CURL=OFF -DGGML_CUDA=ON}"',
    ],
    Path("scripts/run_llamacpp.sh"): [
        "build-cuda/bin/llama-cli",
    ],
    Path("scripts/bench_kv_cache.sh"): [
        "build-cuda/bin/llama-bench",
    ],
    Path("scripts/bench_32k_llamacpp_kv.sh"): [
        "build-cuda/bin/llama-bench",
    ],
    Path("scripts/q2_profile.sh"): [
        "build-cuda/bin/llama-bench",
    ],
    Path("scripts/q2_segment_bench.sh"): [
        "build-cuda/bin/llama-bench",
    ],
    Path("scripts/run_kv_ppl_matrix.sh"): [
        "build-cuda/bin/llama-perplexity",
    ],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def cache_value(cache: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in cache.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1] if "=" in line else ""
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check llama.cpp build/default binary paths.")
    parser.parse_args()

    for rel_path, needles in REQUIRED.items():
        path = ROOT / rel_path
        require(path.exists(), f"missing script: {rel_path}")
        text = path.read_text()
        for needle in needles:
            require(needle in text, f"{rel_path} missing required default: {needle}")

    build_dir = ROOT / "third_party/OSCAR/build-cuda"
    cache_path = build_dir / "CMakeCache.txt"
    require(cache_path.exists(), "expected build-cuda CMakeCache.txt to exist")
    cache = cache_path.read_text(errors="replace")
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_CUDA": "ON",
        "GGML_CUDA_FA": "ON",
        "GGML_CUDA_GRAPHS": "ON",
        "LLAMA_CURL": "OFF",
    }
    for key, expected in expected_cache.items():
        actual = cache_value(cache, key)
        require(actual == expected, f"expected {key}={expected} in build-cuda CMakeCache.txt, got {actual!r}")
    cuda_arch = cache_value(cache, "CMAKE_CUDA_ARCHITECTURES")
    require(cuda_arch is not None and "120" in cuda_arch, f"expected build-cuda CUDA arch to include sm_120, got {cuda_arch!r}")
    require("/cuda-12.9/bin/nvcc" in (cache_value(cache, "CMAKE_CUDA_COMPILER") or ""), "expected build-cuda to use CUDA 12.9 nvcc")

    for name in ("llama-bench", "llama-cli", "llama-perplexity"):
        binary = build_dir / "bin" / name
        require(binary.exists(), f"expected build-cuda {name} to exist")
        require(binary.stat().st_mode & 0o111, f"expected build-cuda {name} to be executable")
    print("build default checks passed")


if __name__ == "__main__":
    main()
