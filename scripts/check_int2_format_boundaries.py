#!/usr/bin/env python3
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    common = (ROOT / "third_party/OSCAR/ggml/src/ggml-common.h").read_text()
    quants = (ROOT / "third_party/OSCAR/ggml/src/ggml-quants.c").read_text()
    triage = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    turbo_notes = (ROOT / "docs/TURBO2_OSCAR_INT2_CUDA_NOTES.md").read_text()

    q2_block = re.search(r"typedef struct \{(?P<body>.*?)\} block_q2_0;", common, re.S)
    require(q2_block is not None, "missing block_q2_0")
    q2_body = q2_block.group("body")
    require("ggml_half d" in q2_body and "ggml_half m" in q2_body and "qs[QK2_0 / 4]" in q2_body,
            "q2_0 must remain the d/m/packed-code block format")
    require("per-block sigma for Lloyd-Max centroids" in q2_body,
            "q2_0 block comments must preserve Lloyd-Max semantics")

    turbo2_block = re.search(r"#define QK_TURBO2.*?typedef struct \{(?P<body>.*?)\} block_turbo2_0;", common, re.S)
    require(turbo2_block is not None, "missing block_turbo2_0")
    turbo2_body = turbo2_block.group("body")
    require("ggml_half norm" in turbo2_body and "qs[QK_TURBO2 / 4]" in turbo2_body,
            "turbo2 must remain norm plus 2-bit indices")
    require("ggml_half m" not in turbo2_body and "zero" not in turbo2_body.lower(),
            "turbo2 must not be confused with q2_0 mean or affine zero metadata")

    turbo3_block = re.search(r"#define QK_TURBO3.*?typedef struct \{(?P<body>.*?)\} block_turbo3_0;", common, re.S)
    require(turbo3_block is not None, "missing block_turbo3_0")
    turbo3_body = turbo3_block.group("body")
    require("ggml_half norm" in turbo3_body and "qs[QK_TURBO3 / 4]" in turbo3_body and "signs[QK_TURBO3 / 8]" in turbo3_body,
            "turbo3 must remain norm plus low bits and high/sign bits")

    for needle in (
        "TURBO2_CENTROIDS[4]",
        "TURBO3_CENTROIDS[8]",
        "quantize_row_turbo2_0_ref",
        "quantize_row_turbo3_0_ref",
        "dequantize_row_turbo2_0",
        "dequantize_row_turbo3_0",
    ):
        require(needle in quants, f"missing TurboQuant local implementation marker: {needle}")

    require("Reference int2 format boundary" in triage,
            "triage report must document the external-reference int2 format boundary")
    require("asymmetric scale/zero" in triage,
            "triage report must say the external-reference int2 uses asymmetric scale/zero metadata")
    require("not llama.cpp `q2_0`" in triage,
            "triage report must explicitly say the reference int2 is not llama.cpp q2_0")
    require("`turbo2/turbo2` is a separate dedicated KV type" in triage,
            "triage report must keep Turbo2 separate from exact q2_0 INT2")

    require("TurboQuant Reference Findings" in turbo_notes,
            "Turbo notes must keep TurboQuant reference section")
    require("not \"plain q3_K KV cache\"" in turbo_notes,
            "Turbo notes must document that q3_K is not the KV route")
    require("Run correctness/PPL checks before calling Turbo2 a replacement for OSCAR INT2" in turbo_notes,
            "Turbo notes must not call Turbo2 a completed OSCAR INT2 replacement")

    print("int2 format boundary checks passed")


if __name__ == "__main__":
    main()
