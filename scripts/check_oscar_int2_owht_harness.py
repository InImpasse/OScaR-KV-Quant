#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    bench = (ROOT / "scripts/bench_32k_llamacpp_kv.sh").read_text()
    server_eval = (ROOT / "scripts/run_gpqa_gsm8k_kv_eval.sh").read_text()
    cli_eval = (ROOT / "scripts/run_gpqa_gsm8k_cli_eval.py").read_text()

    for text, label in ((bench, "32k bench"), (server_eval, "server eval")):
        require('local q2_owht="0"' in text, f"{label} must default q2 OWHT off")
        require('if [[ "$cache_k/$cache_v" == "q2_0/q2_0" && "$no_hadamard" == "1" ]]; then' in text,
                f"{label} must enable q2 OWHT only for OSCAR-style q2_0/q2_0")
        require('LLAMA_KV_Q2_0_OWHT="$q2_owht"' in text,
                f"{label} must pass the computed q2 OWHT gate")
    require('case_enabled oscar_int2' in bench and '"q2_0" "q2_0" "1" "0"' in bench,
            "32k bench oscar_int2 must use no-Hadamard q2 without split clipping")

    oscar_env_block = cli_eval.split("def oscar_env", 1)[1].split("def oscar_q2_env", 1)[0]
    oscar_q2_env_block = cli_eval.split("def oscar_q2_env", 1)[1].split("VARIANTS", 1)[0]
    require('"LLAMA_KV_Q2_0_OWHT"' not in oscar_env_block,
            "CLI base oscar_env must not leak q2-only OWHT into INT4/BF16/Turbo controls")
    for key in ("LLAMA_KV_CLIP_RATIO", "LLAMA_KV_CLIP_RATIO_K", "LLAMA_KV_CLIP_RATIO_V"):
        require(f'"{key}": "0"' in oscar_env_block,
                "CLI OSCAR env must default to no q2 split clipping after writer A/B")
    require('env["LLAMA_KV_Q2_0_OWHT"] = "1"' in oscar_q2_env_block,
            "CLI oscar_q2_env must enable staged q2 OWHT writer for OSCAR int2")
    require('"env": oscar_q2_env()' in cli_eval,
            "CLI oscar q2 variants must use oscar_q2_env")
    require('"LLAMA_KV_Q2_0_OWHT": "0"' in cli_eval.split("def run_cli", 1)[1],
            "CLI runner must keep q2 OWHT off unless a variant opts in")

    print("oscar int2 OWHT harness checks passed")


if __name__ == "__main__":
    main()
