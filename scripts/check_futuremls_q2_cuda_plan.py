#!/usr/bin/env python3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OSCAR = ROOT / "third_party/OSCAR"
PLAN = ROOT / "docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md"
LOCAL_CUDA = OSCAR / "ggml/src/ggml-cuda/fattn-q2_0-f16.cu"
TILE_CUDA = OSCAR / "ggml/src/ggml-cuda/fattn-q2-tile-mixed.cu"
FATTN_VEC_CUDA = OSCAR / "ggml/src/ggml-cuda/fattn-vec.cuh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"futuremls/zhongzhu/llamacpp:{path}"],
        cwd=OSCAR,
        text=True,
    )


def main() -> None:
    plan = PLAN.read_text()
    for marker in [
        "kernel_flash_attn_mixed_mm_q2_0_f16_d128",
        "Q=8",
        "C=64",
        "dedicated D=128 q2/q2 prefill kernel",
        "Do not spend more effort on small changes inside the current q2 vec dot path",
        "run_q2_ramp_next.py",
        "ACK_Q2_RAMP_GATE_HOLD=1",
    ]:
        require(marker in plan, f"plan missing marker: {marker}")
    require(
        "External server-runtime results" in plan and "not an implementation direction" in plan,
        "plan should keep external server-runtime results as references only",
    )

    metal_ops = git_show("ggml/src/ggml-metal/ggml-metal-ops.cpp")
    metal_shader = git_show("ggml/src/ggml-metal/ggml-metal.metal")
    cpu_ops = git_show("ggml/src/ggml-cpu/ops.cpp")
    require("kernel_flash_attn_mixed_mm_q2_0_f16_d128" in metal_ops, "FutureMLS Metal ops should dispatch tiled q2 mixed kernel")
    require("kernel_flash_attn_mixed_mm" in metal_shader, "FutureMLS Metal shader should define tiled mixed kernel")
    require("mm_mixed_pass" in metal_shader, "FutureMLS Metal shader should share online-softmax pass")
    require("Q  = 8" in metal_shader and "C  = 64" in metal_shader, "FutureMLS tiled kernel should use Q=8/C=64")
    require("ggml_compute_forward_flash_attn_ext_mixed" in cpu_ops, "FutureMLS CPU should include mixed attention reference")

    local_cuda = LOCAL_CUDA.read_text()
    tile_cuda = TILE_CUDA.read_text()
    fattn_vec = FATTN_VEC_CUDA.read_text()
    require("flash_attn_q2_tile_mixed_mm_kernel" in tile_cuda or "flash_attn_q2_tile_kernel" in tile_cuda,
            "local CUDA should expose FutureMLS-style q2 tile kernel skeleton")
    require("constexpr int Q_TILE  = 8" in tile_cuda,
            "q2 tile mixed CUDA should keep Q=8")
    require("constexpr int C_TILE  = 64" in tile_cuda,
            "q2 tile mixed CUDA should keep C=64")
    require(
        "for (int j = kv_begin; j < kv_end; ++j)" in local_cuda,
        "local mixed CUDA helper should still be recognized as per-KV scalar fallback path",
    )
    require("LLAMA_TURBO_VEC_STREAM_K" in fattn_vec, "current Turbo2 CUDA fast path should remain env-gated")
    require("launch_fattn<D, cols_per_block, 1>" in fattn_vec and "stream_k" in fattn_vec, "Turbo2 fast path should reuse llama.cpp stream-k launch")

    print("FutureMLS q2 CUDA port plan checks passed")


if __name__ == "__main__":
    main()
