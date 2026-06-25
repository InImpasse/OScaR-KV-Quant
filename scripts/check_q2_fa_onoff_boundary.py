#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/q2_fa_onoff_cli_smoke_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def row_for(rows: list[dict[str, str]], variant: str, dataset: str) -> dict[str, str]:
    matches = [row for row in rows if row["variant"] == variant and row["dataset"] == dataset]
    require(len(matches) == 1, f"expected one row for {variant}/{dataset}, got {len(matches)}")
    return matches[0]


def main() -> None:
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("Flash-attention on/off boundary" in report,
            "triage report must document q2 FA on/off boundary")
    require("V cache quantization requires flash_attn" in report,
            "triage report must record the llama.cpp V-cache quantization guard")

    fa_on_summary = RUN / "fa_on/summary.csv"
    fa_off_summary = RUN / "fa_off/summary.csv"
    require(fa_on_summary.is_file(), "missing q2 FA-on summary")
    require(fa_off_summary.is_file(), "missing q2 FA-off summary")

    with fa_on_summary.open(newline="") as f:
        fa_on_rows = list(csv.DictReader(f))
    for variant in ("plain_int2", "oscar_int2"):
        for dataset in ("gpqa", "gsm8k"):
            row = row_for(fa_on_rows, variant, dataset)
            require(row["correct"] == "0" and row["total"] == "3",
                    f"FA-on {variant}/{dataset} should record 0/3 q2 failure")

    with fa_off_summary.open(newline="") as f:
        fa_off_rows = list(csv.DictReader(f))
    for variant in ("plain_int2", "oscar_int2"):
        for dataset in ("gpqa", "gsm8k"):
            row = row_for(fa_off_rows, variant, dataset)
            require(row["correct"] == "0" and row["total"] == "0",
                    f"FA-off {variant}/{dataset} should remain an unsupported 0/0 control")

    raw = RUN / "fa_off/raw"
    for path in raw.glob("*.json"):
        data = json.loads(path.read_text())
        for case in data["task_states"]["cases"].values():
            require(case["grader_log"]["returncode"] == 1,
                    f"FA-off q2 full-control case should fail context creation: {path.name}")

    print("q2 FA on/off boundary checks passed")


if __name__ == "__main__":
    main()
