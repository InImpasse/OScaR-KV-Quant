#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mixed_vec_smoke_current"
RUNNER = ROOT / "scripts/run_mixed_vec_smoke.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    text = RUNNER.read_text()
    require('DRY_RUN="${DRY_RUN:-1}"' in text, "mixed vec smoke must default to DRY_RUN=1")
    require("ACK_MIXED_VEC_SMOKE" in text, "mixed vec smoke must require ACK_MIXED_VEC_SMOKE")
    require("default_mixed" in text and "mixed_vec" in text and "plain_int2" in text,
            "mixed vec smoke must cover default fused, mixed vec, and pure q2 paths")
    require("LLAMA_KV_MIXED_VEC_MAIN=1" in text, "mixed vec smoke must enable mixed vec env on experimental path")
    require("oscar2_int2_mixed_vec" in (ROOT / "scripts/run_gpqa_gsm8k_cli_eval.py").read_text(),
            "CLI eval must expose oscar2_int2_mixed_vec variant")

    dry = subprocess.run([str(RUNNER)], cwd=ROOT, text=True, capture_output=True, check=False)
    require(dry.returncode == 0, f"default dry-run failed: {dry.stderr}")
    require("Dry run complete" in dry.stdout, "runner should finish dry-run cleanly")
    require("default_mixed" in dry.stdout and "mixed_vec" in dry.stdout, "dry-run should print all smoke paths")

    require((RUN / "config.txt").is_file(), "missing archived mixed vec smoke config")
    require((RUN / "direct/summary.txt").is_file(), "missing archived direct smoke summary")
    for label in ("default_mixed", "mixed_vec", "plain_int2", "oscar_int2"):
        require((RUN / f"direct/{label}.stdout.txt").is_file(), f"missing direct stdout for {label}")
        summary = (RUN / "direct/summary.txt").read_text()
        require(f"{label} rc=0" in summary, f"{label} direct smoke must succeed")

    quality_csv = RUN / "quality/summary.csv"
    if quality_csv.is_file():
        rows = list(csv.DictReader(quality_csv.open()))
        for variant in ("oscar_int2_mixed", "oscar2_int2_mixed_vec", "plain_int2", "oscar_int2", "baseline_bf16"):
            matches = [r for r in rows if r["variant"] == variant]
            require(len(matches) == 2, f"expected gpqa+gsm8k rows for {variant}, got {len(matches)}")
    else:
        raw_dir = RUN / "quality/raw"
        require(raw_dir.is_dir(), "missing quality raw dir or summary.csv")
        for variant in ("oscar_int2_mixed", "oscar2_int2_mixed_vec"):
            for dataset in ("gpqa", "gsm8k"):
                require((raw_dir / f"{variant}_{dataset}.json").is_file(),
                        f"missing quality raw json for {variant}/{dataset}")

    fattn_vec = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-vec.cuh").read_text()
    require("__half22float2(V_h2" in fattn_vec,
            "mixed vec HP V path should use half2 vectorized load")

    print("mixed vec smoke checks passed")


if __name__ == "__main__":
    main()
