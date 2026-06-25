#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fattn = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn.cu").read_text()
    tile = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-tile.cuh").read_text()
    header = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn.cuh").read_text()
    dispatcher = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
    report = (ROOT / "docs/MIXED_K_Q4_V_Q2_QUALITY.md").read_text()

    require(
        "K->type == GGML_TYPE_Q2_0 || V->type == GGML_TYPE_Q2_0" in fattn
        and "K->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO2_0" in fattn
        and "K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0" in fattn
        and "return BEST_FATTN_KERNEL_VEC" in fattn,
        "q2/turbo KV types must remain confined to vector FA until tile is quant-aware",
    )
    require(
        "const half2 * K_h2" in tile and "const half2 * V_h2" in tile,
        "tile FA still assumes half2 K/V loads and must not receive quantized KV tensors",
    )
    require(
        "quant-aware tile load" in report and "half2" in report,
        "mixed-KV report must document why simple tile/MMA dispatch is unsafe",
    )
    require(
        "ggml_cuda_flash_attn_ext_q4_0_turbo3" not in header,
        "negative q4/turbo3 prototype must not expose a CUDA FA interface",
    )
    require(
        not (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-q4_0-turbo3.cu").exists(),
        "negative q4/turbo3 prototype source must stay out of the CUDA glob build",
    )
    require(
        "ggml_cuda_flash_attn_ext_q4_0_turbo3_supported(ctx.device, dst)" not in dispatcher
        and "ggml_cuda_flash_attn_ext_q4_0_turbo3_supported(dev_ctx->device, op)" not in dispatcher,
        "negative q4/turbo3 prototype must not be wired into the CUDA dispatcher",
    )
    print("mixed tile boundary checks passed")


if __name__ == "__main__":
    main()
