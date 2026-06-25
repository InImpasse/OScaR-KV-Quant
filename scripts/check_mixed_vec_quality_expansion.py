#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mixed_vec_quality_expansion.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    text = RUNNER.read_text()
    require('DRY_RUN="${DRY_RUN:-1}"' in text, "quality expansion must default to DRY_RUN=1")
    require("ACK_QUALITY_EXPANSION" in text, "quality expansion must require ACK")
    require("MIN_8K_PP" in text, "quality expansion must gate on 8K pp threshold")
    require("Skipping PPL matrix: CORPUS unset." in text or "run_kv_ppl_matrix.sh" in text,
            "quality expansion must document PPL path")
    require("oscar2_int2_mixed_vec" in text, "quality expansion must include mixed vec variant")
    require("8192" in text, "quality expansion should use long-context eval")
    print("mixed vec quality expansion checks passed")


if __name__ == "__main__":
    main()
