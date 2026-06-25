#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "runs/oscar_turbo3_cli_smoke_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def row_for(rows: list[dict[str, str]], variant: str, dataset: str) -> dict[str, str]:
    matches = [row for row in rows if row["variant"] == variant and row["dataset"] == dataset]
    require(len(matches) == 1, f"expected one row for {variant}/{dataset}, got {len(matches)}")
    return matches[0]


def main() -> None:
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("Turbo3 CLI smoke" in report, "triage report must include Turbo3 CLI smoke")
    require("not a" in report and "validated quality replacement for OSCAR INT2" in report,
            "triage report must not present Turbo3 as a quality replacement")

    summary = SMOKE / "summary.csv"
    require(summary.is_file(), "missing oscar_turbo3 smoke summary.csv")
    with summary.open(newline="") as f:
        rows = list(csv.DictReader(f))

    require(row_for(rows, "baseline_bf16", "gpqa")["correct"] == "1", "baseline GPQA smoke should retain known result")
    require(row_for(rows, "baseline_bf16", "gsm8k")["correct"] == "1", "baseline GSM8K smoke should retain known result")
    require(row_for(rows, "plain_int3", "gpqa")["correct"] == "1", "plain_int3 GPQA smoke should retain known result")
    require(row_for(rows, "plain_int3", "gsm8k")["correct"] == "0", "plain_int3 GSM8K smoke should record failure")
    require(row_for(rows, "oscar_turbo3", "gpqa")["correct"] == "0", "oscar_turbo3 GPQA smoke should record failure")
    require(row_for(rows, "oscar_turbo3", "gsm8k")["correct"] == "0", "oscar_turbo3 GSM8K smoke should record failure")

    raw = SMOKE / "raw"
    oscar_gsm8k = json.loads((raw / "oscar_turbo3_gsm8k.json").read_text())
    plain_gsm8k = json.loads((raw / "plain_int3_gsm8k.json").read_text())
    oscar_text = "\n".join(case["response"] for case in oscar_gsm8k["task_states"]["cases"].values())
    plain_text = "\n".join(case["response"] for case in plain_gsm8k["task_states"]["cases"].values())
    require("So much. So much." in oscar_text or "\\] \\] \\]" in oscar_text,
            "oscar_turbo3 raw output should preserve the observed repeated-token failure")
    require("````````" in plain_text or "[end of text]" in plain_text,
            "plain_int3 raw output should preserve the observed empty/repeated output")

    print("oscar_turbo3 smoke checks passed")


if __name__ == "__main__":
    main()
