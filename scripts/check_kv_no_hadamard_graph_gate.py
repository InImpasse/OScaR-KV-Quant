#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> None:
    kv_cache = (ROOT / "third_party/OSCAR/src/llama-kv-cache.cpp").read_text()
    q2_owht = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/q2_0-owht.cuh").read_text()

    require("LLAMA_KV_NO_HADAMARD" in q2_owht and "q2_0_cuda_apply_hadamard" in q2_owht,
            "q2 OWHT writer must keep the no-Hadamard gate")
    require("LLAMA_KV_NO_HADAMARD" in kv_cache,
            "KV graph attention rotation must observe LLAMA_KV_NO_HADAMARD")
    require("LLAMA_ATTN_ROT_DISABLE" in kv_cache,
            "legacy attention-rotation disable env must remain supported")
    require("attention Hadamard rotation force disabled" in kv_cache,
            "KV graph must log when attention Hadamard rotation is disabled")
    require("attn_rot_disable" in kv_cache and "LLAMA_KV_NO_HADAMARD" in kv_cache.split("attn_rot_disable", 1)[1].split("if (attn_rot_disable)", 1)[0],
            "LLAMA_KV_NO_HADAMARD must feed attn_rot_disable before attn_rot_k/v are computed")

    print("KV no-Hadamard graph gate checks passed")


if __name__ == "__main__":
    main()
