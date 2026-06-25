#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "scripts/report_recovery_readiness.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    output = subprocess.check_output([sys.executable, str(REPORT), "--no-gpu"], text=True)
    require("overall_status=complete" in output, "readiness report should expose deliverable complete status")
    require("exact_int2_research_status=incomplete" in output, "readiness report should expose exact INT2 research status")
    require("can_mark_complete=true" in output, "readiness report should expose completion gate")
    require("Turbo2 is a separate reference" in output, "readiness report should separate Turbo2 from exact INT2")
    require("gpu_snapshot=" in output, "readiness report should include a GPU snapshot")
    require("skipped (--no-gpu)" in output, "readiness check should skip GPU snapshot")
    require("archive_checksums_ok=true" in output, "readiness report should validate archive checksums")
    require("Archive checksums" in output, "readiness report should include archive checksum section")
    require("matrix: ok=true" in output, "readiness report should validate matrix archive")
    require("cuda_graph_ab: ok=true" in output, "readiness report should validate graph archive")
    require("goal_status: ok=true" in output, "readiness report should validate goal-status archive")
    require("q2_cuda_path: ok=true" in output, "readiness report should validate q2 path archive")
    require("q2_cuda_path_archive_fresh=true" in output, "readiness report should show q2 path archive freshness")
    require("q2_kq_dp4a_calls=3" in output, "readiness report should show q2 KQ dp4a count")
    require("q2_ramp_recommendation=hold_32k_q2" in output, "readiness report should hold known-failed 32k q2")
    require("q2_ramp_next_prompt=512" in output, "readiness report should show first missing q2 recovery step")
    require("q2_ramp_failed_32k=true" in output, "readiness report should flag failed 32k q2")
    require("Q2 ramp gate" in output, "readiness report should include q2 ramp gate section")
    require("print_32k_q2_ramp_commands.py" in output, "readiness report should show q2 ramp dry-run helper")
    require("print_32k_matrix_commands.py" in output, "readiness report should show matrix dry-run helper")
    require("Recommended next actions" in output, "readiness report should include recommended actions")
    require("--real --ack-real" in output, "readiness report should document real-command ACK")
    require("--ack-q2-ramp-gate-hold" in output, "readiness report should document ramp gate hold ACK")
    require("llama-bench" not in REPORT.read_text(), "readiness script must not launch benchmark commands")

    json_output = subprocess.check_output([sys.executable, str(REPORT), "--json", "--no-gpu"], text=True)
    data = json.loads(json_output)
    require(data["overall_status"] == "complete", "JSON readiness should expose deliverable complete status")
    require(data["exact_int2_research_status"] == "incomplete", "JSON readiness should expose exact INT2 research status")
    require(data["completion_gate"]["can_mark_complete"] is True, "JSON completion gate should be true for current deliverable")
    require(data["completion_gate"]["blocking_items"] == [], "JSON completion gate should not block the current INT4 deliverable")
    require(data["completion_gate"]["required_missing_evidence"] == [], "JSON completion gate should not require exact INT2 evidence for current deliverable")
    require("gpu_snapshot" in data, "JSON readiness should include GPU snapshot")
    require(data["gpu_snapshot"] == "skipped (--no-gpu)", "JSON readiness check should skip GPU snapshot")
    archive_names = {row["name"] for row in data["archives"]}
    require(
        archive_names == {"matrix", "cuda_graph_ab", "goal_status", "q2_cuda_path"},
        "JSON readiness should include all key run archives",
    )
    require(all(row["ok"] is True for row in data["archives"]), "JSON readiness should validate all archive checksums")
    require(all(row["files"] > 0 for row in data["archives"]), "JSON readiness archives should contain files")
    require(
        all(not row["missing"] and not row["mismatched"] and not row["malformed"] for row in data["archives"]),
        "JSON readiness archives should have no checksum failures",
    )
    require(data["q2_cuda_path"]["fresh"] is True, "JSON readiness should confirm q2 CUDA path archive is fresh")
    require(data["q2_cuda_path"]["q2_kq_dp4a_calls"] == "3", "JSON readiness should include q2 KQ dp4a count")
    require(data["q2_cuda_path"]["q4_kq_dp4a_calls"] == "1", "JSON readiness should include q4 KQ dp4a count")
    require(data["q2_ramp_gate"]["recommendation"] == "hold_32k_q2", "JSON readiness should hold 32k q2")
    require(data["q2_ramp_gate"]["next_prompt"] == 512, "JSON readiness should show first missing q2 recovery step")
    require(data["q2_ramp_gate"]["failed_32k_q2"] is True, "JSON readiness should flag failed 32k q2")
    require(
        data["q2_ramp_gate"]["ramp_prompts"] == [512, 2048, 4096, 8192, 16384, 32768],
        "JSON readiness should include full q2 recovery ramp",
    )
    require(
        any(row["item"] == "32k_int2_speed_target" and row["status"] == "incomplete" for row in data["incomplete_items"]),
        "JSON readiness should include incomplete exact 32k INT2 item",
    )
    require(
        any(row["item"] == "recovery_readiness_report" and row["status"] == "complete" for row in data["guardrails"]),
        "JSON readiness should include readiness guardrail",
    )
    require(
        any("--real --ack-real" in command for command in data["real_command_printing"]),
        "JSON readiness should include ACK-protected real command printing",
    )
    require(
        any("--ack-32k-q2-real" in command for command in data["real_command_printing"]),
        "JSON readiness should document separate 32k q2 real ACK",
    )
    require(
        any("--ack-q2-ramp-gate-hold" in command for command in data["real_command_printing"]),
        "JSON readiness should document separate ramp gate hold ACK",
    )
    require(
        any("Turbo2 is a separate reference" in action for action in data["recommended_next_actions"]),
        "JSON readiness should separate Turbo2 from exact INT2 in recommended actions",
    )
    require(
        any("512/2k/4k/8k/16k q2 ramp" in action for action in data["recommended_next_actions"]),
        "JSON readiness should recommend crash-recovery q2 ramp before 32k q2",
    )

    print("recovery readiness report checks passed")


if __name__ == "__main__":
    main()
