#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    cuda_owht = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/q2_0-owht.cuh").read_text()
    cpu_quant = (ROOT / "third_party/OSCAR/ggml/src/ggml-quants.c").read_text()
    common = (ROOT / "third_party/OSCAR/ggml/src/ggml-common.h").read_text()

    require("for (int ib = 0; ib < actual_nb; ++ib)" in cuda_owht,
            "CUDA q2 OWHT writer must loop over all blocks")
    require("dst[ib].m = mean_h;" in cuda_owht,
            "CUDA q2 OWHT writer must replicate group mean to every block")
    require("dst[ib].m = __float2half(0.0f)" not in cuda_owht,
            "CUDA q2 OWHT writer must not zero non-first block means")
    require("q2_0_cuda_clip_ratio_for_cache" in cuda_owht,
            "CUDA q2 OWHT path must expose K/V-specific clip ratio selection")
    require("LLAMA_KV_CLIP_RATIO_K" in cuda_owht and "LLAMA_KV_CLIP_RATIO_V" in cuda_owht,
            "CUDA q2 OWHT path must support reference OSCAR K/V clip ratios")
    set_rows = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/set-rows.cu").read_text()
    require("q2_0_cuda_clip_ratio_for_cache(dst_name)" in set_rows,
            "CUDA set_rows q2 OWHT dispatch must choose clip ratio from destination cache name")
    require("stream, dst->name" in set_rows,
            "CUDA set_rows q2 OWHT dispatch must pass the destination tensor name")
    require("Store the group mean in *every* block" in cpu_quant,
            "CPU q2 reference must document replicated group mean")
    require("OWHT path: group mean in first block, otherwise zero" not in common,
            "block_q2_0 comment must not describe stale first-block-only OWHT mean semantics")

    print("q2 OWHT mean semantics checks passed")


if __name__ == "__main__":
    main()
