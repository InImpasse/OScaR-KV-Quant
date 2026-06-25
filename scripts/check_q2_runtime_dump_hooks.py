#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    graph = (ROOT / "third_party/OSCAR/src/llama-graph.cpp").read_text()
    kv_cache = (ROOT / "third_party/OSCAR/src/llama-kv-cache.cpp").read_text()
    debug = (ROOT / "third_party/OSCAR/common/debug.cpp").read_text()

    require('cb(k_store, "cache_k_set_rows", il);' in graph,
            "llama graph must expose K set_rows output to debug callback")
    require('cb(v_store, "cache_v_set_rows", il);' in graph,
            "llama graph must expose V set_rows output to debug callback")
    require('ggml_format_name(k_cur, "cache_k_set_rows_src_l%d", il);' in kv_cache,
            "KV cache K set_rows source view must have a stable debug name")
    require('ggml_format_name(v_cur, "cache_v_set_rows_src_l%d", il);' in kv_cache,
            "KV cache V set_rows source view must have a stable debug name")
    require("LLAMA_DEBUG_TENSOR_DUMP_DIR" in debug and "common_debug_dump_tensor" in debug,
            "common debug callback must support tensor dumps")
    require("GGML_OP_SET_ROWS" in debug and 'set_rows_srcs[] = { t->src[0], t->src[1] }' in debug,
            "debug callback must dump set_rows source rows and index tensor alongside the cache output")
    require("ggml_backend_tensor_get" in debug,
            "debug callback must copy device tensors before dumping")

    print("q2 runtime dump hook checks passed")


if __name__ == "__main__":
    main()
