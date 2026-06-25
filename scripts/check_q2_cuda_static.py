#!/usr/bin/env python3
import argparse
import itertools
import re
from pathlib import Path


REQUIRED = [
    "Q2_0_FATTN_SIGN_LUT",
    "Q2_0_FATTN_HIGH_LUT",
    "vec_dot_fattn_vec_KQ_q2_0_chunk",
    "const int sum_sign = ggml_cuda_dp4a(sign_i, u, 0);",
    "const int sum_high = ggml_cuda_dp4a(high_i, u, 0);",
    "const int usum     = ggml_cuda_dp4a(Q2_0_FATTN_ONES_I32, u, 0);",
    "d*(Q2_0_LM_C2*sum_sign + (Q2_0_LM_C3 - Q2_0_LM_C2)*sum_high) + m*usum",
    "dequantize_V_q2_0",
    "q2_0_dequantize_scalar_cuda",
]

Q4_REQUIRED = [
    "vec_dot_fattn_vec_KQ_q4_0",
    "const int sumi = ggml_cuda_dp4a(v, u, 0);",
]

FORBIDDEN = [
    "GGML_CUDA_Q2_FATTN_FAST",
    "GGML_CUDA_Q2_FATTN_TILE_D128",
    "ncols_partial",
    "dm[4][2]",
]


def extract_function(text: str, name: str) -> str:
    start = text.find(name)
    if start < 0:
        raise AssertionError(f"missing function: {name}")
    brace = text.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing function body: {name}")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise AssertionError(f"unterminated function body: {name}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_lut(text: str, name: str) -> list[int]:
    match = re.search(rf"{name}\[256\]\s*=\s*\{{(?P<body>.*?)\}};", text, re.S)
    if not match:
        raise AssertionError(f"missing LUT body: {name}")
    vals = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", match.group("body"))]
    require(len(vals) == 256, f"{name} should have 256 entries, got {len(vals)}")
    return vals


def signed_byte(v: int) -> int:
    v &= 0xFF
    return v - 256 if v >= 128 else v


def lane(v: int, i: int) -> int:
    return signed_byte(v >> (8 * i))


def dp4a(a: int, b: int) -> int:
    return sum(lane(a, i) * lane(b, i) for i in range(4))


def check_q2_lut_equivalence(common: str) -> None:
    sign_lut = parse_lut(common, "Q2_0_FATTN_SIGN_LUT")
    high_lut = parse_lut(common, "Q2_0_FATTN_HIGH_LUT")

    # The q2 KQ formula reconstructs Lloyd-Max centroids as:
    # c(q) = C2 * sign(q) + (C3 - C2) * high(q), plus the block mean term.
    # Validate the integer LUTs for every packed q2 byte and a set of signed
    # q8 lane values. This catches byte-order and sign-extension mistakes.
    test_lanes = [-128, -37, -1, 0, 1, 42, 127]
    for packed in range(256):
        signs = []
        highs = []
        codes = []
        for i in range(4):
            q = (packed >> (2 * i)) & 0x03
            codes.append(q)
            signs.append(-1 if q < 2 else 1)
            highs.append(-1 if q == 0 else (1 if q == 3 else 0))
        for qs in itertools.product(test_lanes, repeat=4):
            u = sum(((q & 0xFF) << (8 * i)) for i, q in enumerate(qs))
            sum_sign = dp4a(sign_lut[packed], u)
            sum_high = dp4a(high_lut[packed], u)
            ref_sign = sum(signs[i] * qs[i] for i in range(4))
            ref_high = sum(highs[i] * qs[i] for i in range(4))
            require(sum_sign == ref_sign,
                    f"q2 sign LUT mismatch packed={packed:#04x} qs={qs}: {sum_sign} != {ref_sign}")
            require(sum_high == ref_high,
                    f"q2 high LUT mismatch packed={packed:#04x} qs={qs}: {sum_high} != {ref_high}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check q2 CUDA static guardrails.")
    parser.add_argument(
        "--cuda-dir",
        type=Path,
        default=Path("third_party/OSCAR/ggml/src/ggml-cuda"),
    )
    args = parser.parse_args()

    files = [
        args.cuda_dir / "fattn-common.cuh",
        args.cuda_dir / "fattn-vec.cuh",
        args.cuda_dir / "fattn.cu",
    ]
    for path in files:
        require(path.exists(), f"missing CUDA source: {path}")

    common = files[0].read_text()
    combined = "\n".join(path.read_text() for path in files)

    for needle in REQUIRED:
        require(needle in common, f"missing q2 baseline feature in fattn-common.cuh: {needle}")

    for needle in Q4_REQUIRED:
        require(needle in common, f"missing q4 comparison feature in fattn-common.cuh: {needle}")

    for needle in FORBIDDEN:
        require(needle not in combined, f"forbidden failed q2 experiment still present: {needle}")

    require(common.count("Q2_0_FATTN_SIGN_LUT") >= 2, "q2 sign LUT should be declared and used")
    require(common.count("Q2_0_FATTN_HIGH_LUT") >= 2, "q2 high LUT should be declared and used")
    require(common.count("ggml_cuda_dp4a") >= 3, "q2 baseline should keep dp4a reconstruction path")

    q2_chunk = extract_function(common, "vec_dot_fattn_vec_KQ_q2_0_chunk")
    q4_kq = extract_function(common, "vec_dot_fattn_vec_KQ_q4_0")
    q2_dp4a_count = q2_chunk.count("ggml_cuda_dp4a")
    q4_dp4a_count = q4_kq.count("ggml_cuda_dp4a")
    require(q2_dp4a_count == 3, f"q2 chunk should use exactly 3 dp4a calls, found {q2_dp4a_count}")
    require(q4_dp4a_count == 1, f"q4 KQ should use exactly 1 dp4a call, found {q4_dp4a_count}")
    require("m*usum" in q2_chunk, "q2 chunk must keep exact mean term m*usum")
    require("Q2_0_FATTN_SIGN_LUT" in q2_chunk, "q2 chunk must use sign LUT")
    require("Q2_0_FATTN_HIGH_LUT" in q2_chunk, "q2 chunk must use high LUT")
    require("m*usum" not in q4_kq, "q4 KQ should not include q2 mean term")
    check_q2_lut_equivalence(common)

    print(f"q2 CUDA static checks passed (q2_chunk_dp4a={q2_dp4a_count}, q4_kq_dp4a={q4_dp4a_count})")


if __name__ == "__main__":
    main()
