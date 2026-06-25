#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "scripts/report_q2_ramp_gate.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    text = subprocess.check_output([sys.executable, str(REPORT)], text=True)
    require("q2/int2 ramp gate" in text, "text report should identify the ramp gate")
    require("ramp_prompts=512,2048,4096,8192,16384,32768" in text, "text report should show full cautious ramp")
    require("completed_prompts=16384" in text, "current archive should only prove 16k q2 speed")
    require("failed_32k_q2=true" in text, "text report should flag known 32k q2 failure")
    require("recommendation=hold_32k_q2" in text, "text report should hold 32k q2 after known failure")
    require("Prefer code/profiler work" in text, "text report should recommend code/profiler work")

    json_output = subprocess.check_output([sys.executable, str(REPORT), "--json"], text=True)
    data = json.loads(json_output)
    require(data["ramp_prompts"] == [512, 2048, 4096, 8192, 16384, 32768], "JSON ramp should include cautious prompts")
    require(data["completed_prompts"] == [16384], "JSON ramp should only count archived valid q2 prompts")
    require(data["failed_32k_q2"] is True, "JSON ramp should flag failed 32k q2")
    require(data["recommendation"] == "hold_32k_q2", "JSON ramp should hold 32k q2")
    require(data["next_prompt"] == 512, "JSON ramp should identify first missing recovery step")
    require(
        any(step["prompt"] == 32768 and step["any_failed"] is True for step in data["steps"]),
        "JSON ramp should include failed 32k step",
    )
    require(
        any(step["prompt"] == 16384 and step["any_ok"] is True for step in data["steps"]),
        "JSON ramp should include valid 16k step",
    )
    require("subprocess" not in REPORT.read_text(), "ramp gate report must not execute workload commands")

    print("q2 ramp gate checks passed")


if __name__ == "__main__":
    main()
