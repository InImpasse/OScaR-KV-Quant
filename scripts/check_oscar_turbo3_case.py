#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    bench = (ROOT / "scripts/bench_32k_llamacpp_kv.sh").read_text()
    printer = (ROOT / "scripts/print_32k_matrix_commands.py").read_text()
    checker = (ROOT / "scripts/check_32k_matrix_commands.py").read_text()
    cli_eval = (ROOT / "scripts/run_gpqa_gsm8k_cli_eval.py").read_text()
    server_eval = (ROOT / "scripts/run_gpqa_gsm8k_kv_eval.sh").read_text()
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()

    require('case_enabled oscar_turbo3 && run_case "oscar_turbo3_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$OSCAR_MODEL" "turbo3" "turbo3" "1" "0.96" "0"' in bench,
            "32k harness must map oscar_turbo3 to rotated GGUF plus turbo3/turbo3 without Stream-K")
    require('("oscar_turbo3", "180", {})' in printer,
            "matrix command printer must include oscar_turbo3 as a non-q2 ACK case")
    require('"oscar_turbo3"' in checker and "expected eleven benchmark commands" in checker,
            "matrix command checker must include oscar_turbo3")

    oscar_turbo3_block = cli_eval.split('"oscar_turbo3"', 1)[1].split('"plain_turbo3"', 1)[0]
    require('granite-4.0-1b-base-bf16-rot-kv.gguf' in oscar_turbo3_block,
            "CLI eval oscar_turbo3 must use the rotated GGUF")
    require('"cache_k": "turbo3"' in oscar_turbo3_block and '"cache_v": "turbo3"' in oscar_turbo3_block,
            "CLI eval oscar_turbo3 must use turbo3/turbo3")
    require('"plain_int3"' in cli_eval and '"plain_turbo3"' in cli_eval,
            "CLI eval must expose both plain_int3 alias and plain_turbo3")

    require('case_enabled oscar_turbo3 && run_variant oscar_turbo3 "$OSCAR_MODEL" turbo3 turbo3 1 0.96 0' in server_eval,
            "server eval must include oscar_turbo3")
    require('case_enabled plain_int3 && run_variant plain_int3 "$BASE_MODEL" turbo3 turbo3 0 0 0' in server_eval,
            "server eval must include plain_int3")
    require("`oscar_turbo3` is now a first-class eval/32k harness case" in report,
            "triage report must document the oscar_turbo3 probe")

    print("oscar_turbo3 case checks passed")


if __name__ == "__main__":
    main()
