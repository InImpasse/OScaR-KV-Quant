#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


REQUIRED = {
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_BF16",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_IQ4_NL",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q2_0",
    "GGML_TYPE_TURBO2_0",
    "GGML_TYPE_TURBO3_0",
}

FORBIDDEN_3BIT = {
    "GGML_TYPE_Q3_K",
    "GGML_TYPE_IQ3_XXS",
    "GGML_TYPE_IQ3_XS",
    "GGML_TYPE_IQ3_S",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_kv_cache_types(text: str) -> set[str]:
    match = re.search(r"const std::vector<ggml_type> kv_cache_types = \{(?P<body>.*?)\};", text, re.S)
    if not match:
        raise AssertionError("could not find kv_cache_types declaration")
    return set(re.findall(r"GGML_TYPE_[A-Z0-9_]+", match.group("body")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check llama.cpp KV cache type registration.")
    parser.add_argument("--arg-cpp", type=Path, default=Path("third_party/OSCAR/common/arg.cpp"))
    args = parser.parse_args()

    text = args.arg_cpp.read_text()
    types = parse_kv_cache_types(text)

    missing = sorted(REQUIRED - types)
    forbidden = sorted(FORBIDDEN_3BIT & types)
    require(not missing, f"missing expected KV cache types: {', '.join(missing)}")
    require(not forbidden, f"3-bit weight-only types should not be exposed as KV cache types: {', '.join(forbidden)}")
    require("GGML_TYPE_Q2_0" in types and "GGML_TYPE_Q4_0" in types and "GGML_TYPE_BF16" in types,
            "matrix cache types must remain exposed")
    require("GGML_TYPE_TURBO3_0" in types,
            "plain_int3 is expected to map to the TurboQuant 3-bit KV cache type turbo3")

    pretty = ", ".join(sorted(types))
    print(f"KV cache type checks passed: {pretty}")


if __name__ == "__main__":
    main()
