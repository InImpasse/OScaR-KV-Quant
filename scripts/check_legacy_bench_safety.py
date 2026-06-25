#!/usr/bin/env python3
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    ROOT / "scripts/bench_kv_cache.sh": {
        "binary": "build-cuda/bin/llama-bench",
        "ack": 'ACK_HEAVY_CONTEXT="${ACK_HEAVY_CONTEXT:-0}"',
        "refusal": "Refusing heavy KV benchmark without ACK_HEAVY_CONTEXT=1",
        "binary_check": 'if [[ ! -x "$LLAMA_BENCH" ]]',
    },
    ROOT / "scripts/bench_kv_cache_matrix.sh": {
        "binary": "build-cuda/bin/llama-bench",
        "ack": 'ACK_MATRIX_BENCH="${ACK_MATRIX_BENCH:-0}"',
        "refusal": "Refusing KV matrix benchmark without ACK_MATRIX_BENCH=1",
        "binary_check": 'if [[ ! -x "$LLAMA_BENCH" ]]',
    },
    ROOT / "scripts/run_kv_ppl_matrix.sh": {
        "binary": "build-cuda/bin/llama-perplexity",
        "ack": 'ACK_PPL_MATRIX="${ACK_PPL_MATRIX:-0}"',
        "refusal": "Refusing PPL matrix without ACK_PPL_MATRIX=1",
        "binary_check": 'if [[ ! -x "$LLAMA_PPL" ]]',
    },
    ROOT / "scripts/bench_32k_llamacpp_kv.sh": {
        "binary": "build-cuda/bin/llama-bench",
        "ack": 'DRY_RUN="${DRY_RUN:-1}"',
        "refusal": "Dry run complete; no results written.",
        "binary_check": 'if [[ ! -x "$LLAMA_BENCH" ]]',
        "dry_only": True,
        "extra_markers": [
            'ACK_Q2_RAMP_GATE_HOLD="${ACK_Q2_RAMP_GATE_HOLD:-0}"',
            "Refusing real 32k q2/int2 while q2 ramp gate is hold_32k_q2",
        ],
    },
    ROOT / "scripts/run_llamacpp.sh": {
        "binary": "build-cuda/bin/llama-cli",
        "ack": 'ACK_RUN_LLAMA="${ACK_RUN_LLAMA:-0}"',
        "refusal": "Refusing llama-cli inference without ACK_RUN_LLAMA=1",
        "binary_check": 'if [[ ! -x "$LLAMA_CLI" ]]',
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check legacy KV bench safety defaults.")
    parser.parse_args()

    for script, expected in SCRIPTS.items():
        text = script.read_text()
        name = script.relative_to(ROOT)
        require(expected["binary"] in text, f"{name} must use build-cuda binary by default")
        require('DRY_RUN="${DRY_RUN:-1}"' in text, f"{name} must default to DRY_RUN=1")
        require(expected["ack"] in text, f"{name} must require an explicit ACK")
        require(expected["refusal"] in text, f"{name} missing ACK refusal message")
        require("Dry run complete;" in text, f"{name} dry-run must avoid real execution")
        require(
            text.index('if [[ "$DRY_RUN" == "1" ]]') < text.index(expected["binary_check"]),
            f"{name} dry-run must render commands before executable checks",
        )
        for marker in expected.get("extra_markers", []):
            require(marker in text, f"{name} missing marker: {marker}")
        if not expected.get("dry_only"):
            require(
                text.index(expected["refusal"]) < text.index(expected["binary_check"]),
                f"{name} ACK guard must run before executable checks",
            )
    print("legacy bench safety checks passed")


if __name__ == "__main__":
    main()
