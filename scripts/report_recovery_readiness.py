#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import report_q2_cuda_path
import report_q2_ramp_gate


ROOT = Path(__file__).resolve().parents[1]
GOAL_STATUS = ROOT / "runs/goal_status_current/goal_status.csv"
Q2_CUDA_PATH_ARCHIVE = ROOT / "runs/q2_cuda_path_current/q2_cuda_path.csv"
ARCHIVES = {
    "matrix": ROOT / "runs/llamacpp_32k_kv_matrix_current/SHA256SUMS",
    "cuda_graph_ab": ROOT / "runs/cuda_graph_ab_512_current/SHA256SUMS",
    "goal_status": ROOT / "runs/goal_status_current/SHA256SUMS",
    "q2_cuda_path": ROOT / "runs/q2_cuda_path_current/SHA256SUMS",
}


def load_goal_rows() -> list[dict[str, str]]:
    with GOAL_STATUS.open(newline="") as f:
        return list(csv.DictReader(f))


def load_q2_cuda_path_archive() -> list[dict[str, str]]:
    with Q2_CUDA_PATH_ARCHIVE.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_checksum_status(name: str, sums_path: Path) -> dict:
    rel_sums = sums_path.relative_to(ROOT).as_posix()
    status = {
        "name": name,
        "sha256sums": rel_sums,
        "ok": False,
        "files": 0,
        "missing": [],
        "mismatched": [],
        "malformed": [],
    }
    if not sums_path.exists():
        status["missing"].append(rel_sums)
        return status

    for lineno, line in enumerate(sums_path.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, rel_path = line.split("  ", 1)
        except ValueError:
            status["malformed"].append(f"{rel_sums}:{lineno}")
            continue
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            status["malformed"].append(f"{rel_sums}:{lineno}")
            continue

        path = ROOT / rel_path
        status["files"] += 1
        if not path.exists():
            status["missing"].append(rel_path)
            continue
        actual = sha256(path)
        if actual != expected:
            status["mismatched"].append(rel_path)

    status["ok"] = (
        status["files"] > 0
        and not status["missing"]
        and not status["mismatched"]
        and not status["malformed"]
    )
    return status


def archive_statuses() -> list[dict]:
    return [archive_checksum_status(name, path) for name, path in ARCHIVES.items()]


def q2_cuda_path_status() -> dict:
    current = report_q2_cuda_path.rows_from_source(
        report_q2_cuda_path.DEFAULT_SOURCE.read_text(errors="replace")
    )
    archived = load_q2_cuda_path_archive()
    fresh = current == archived
    q2_kq = next((row for row in current if row["path"] == "KQ" and row["type"] == "q2_0"), {})
    q4_kq = next((row for row in current if row["path"] == "KQ" and row["type"] == "q4_0"), {})
    return {
        "archive": str(Q2_CUDA_PATH_ARCHIVE.relative_to(ROOT)),
        "fresh": fresh,
        "current_rows": len(current),
        "archived_rows": len(archived),
        "q2_kq_dp4a_calls": q2_kq.get("dp4a_calls", ""),
        "q2_kq_fingerprint": q2_kq.get("fingerprint", ""),
        "q4_kq_dp4a_calls": q4_kq.get("dp4a_calls", ""),
        "q4_kq_fingerprint": q4_kq.get("fingerprint", ""),
    }


def gpu_snapshot() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi unavailable"
    # Keep this report read-only: one short snapshot, no benchmark/profiler launch.
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return f"nvidia-smi failed: {result.stderr.strip()}"
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "nvidia-smi returned no rows"


def build_report(no_gpu: bool = False) -> dict:
    rows = load_goal_rows()
    by_item = {row["item"]: row for row in rows}
    overall_status = by_item.get("overall_status", {}).get("status", "missing")
    research_incomplete = [
        row for row in rows
        if row["status"] != "complete" and row["item"] != "overall_status"
    ]
    blocking_items = [] if overall_status == "complete" else [row["item"] for row in research_incomplete]
    guardrail_items = [
        "llamacpp_only_guardrails",
        "execution_safety_guardrails",
        "recovery_command_guardrails",
        "post_case_cooldown_guard",
        "recovery_readiness_report",
        "q2_cuda_static_guardrails",
        "q2_cuda_path_archive",
    ]
    return {
        "overall_status": overall_status,
        "exact_int2_research_status": by_item.get("exact_int2_research_status", {}).get("status", "missing"),
        "completion_gate": {
            "can_mark_complete": overall_status == "complete",
            "blocking_items": blocking_items,
            "required_missing_evidence": [
                "valid exact 32k q2_0/q2_0 OSCAR INT2 llama.cpp speed result"
            ] if overall_status != "complete" and "32k_int2_speed_target" in blocking_items else [],
        },
        "gpu_snapshot": "skipped (--no-gpu)" if no_gpu else gpu_snapshot(),
        "archives": archive_statuses(),
        "q2_ramp_gate": report_q2_ramp_gate.build_report(
            report_q2_ramp_gate.DEFAULT_MATRIX
        ),
        "q2_cuda_path": q2_cuda_path_status(),
        "incomplete_items": research_incomplete,
        "guardrails": [
            {
                "item": item,
                "status": by_item.get(item, {"status": "missing"})["status"],
                "note": by_item.get(item, {"note": ""}).get("note", ""),
            }
            for item in guardrail_items
        ],
        "dry_run_commands": [
            "python3 scripts/print_32k_q2_ramp_commands.py",
            "python3 scripts/print_32k_matrix_commands.py",
        ],
        "recommended_next_actions": [
            "Turbo2 is a separate reference; exact OSCAR INT2 still requires a valid 32k q2_0/q2_0 speed and quality result.",
            "Use dry-run command helpers first; they do not execute benchmarks.",
            "If exact q2_0 testing resumes after a crash, start with the 512/2k/4k/8k/16k q2 ramp before any 32k q2_0 attempt.",
            "Printing 32k q2 real commands requires --ack-32k-q2-real and --ack-q2-ramp-gate-hold in addition to --real --ack-real.",
            "Prefer code/profiler work before repeating the known 32k q2 NO-GO path.",
        ],
        "real_command_printing": [
            "python3 scripts/print_32k_q2_ramp_commands.py --real --ack-real",
            "python3 scripts/print_32k_q2_ramp_commands.py --real --ack-real --ack-32k-q2-real --ack-q2-ramp-gate-hold",
            "python3 scripts/print_32k_matrix_commands.py --real --ack-real",
            "python3 scripts/print_32k_matrix_commands.py --real --ack-real --ack-32k-q2-real --ack-q2-ramp-gate-hold",
        ],
    }


def print_text(report: dict) -> None:
    print("# llama.cpp 32k KV recovery readiness")
    print()
    print(f"overall_status={report['overall_status']}")
    print(f"exact_int2_research_status={report['exact_int2_research_status']}")
    print(f"can_mark_complete={str(report['completion_gate']['can_mark_complete']).lower()}")
    print(f"gpu_snapshot={report['gpu_snapshot']}")
    print(
        "archive_checksums_ok="
        + str(all(row["ok"] for row in report["archives"])).lower()
    )
    print(f"q2_cuda_path_archive_fresh={str(report['q2_cuda_path']['fresh']).lower()}")
    print(f"q2_kq_dp4a_calls={report['q2_cuda_path']['q2_kq_dp4a_calls']}")
    print(f"q2_ramp_recommendation={report['q2_ramp_gate']['recommendation']}")
    print(f"q2_ramp_next_prompt={report['q2_ramp_gate']['next_prompt']}")
    print(f"q2_ramp_failed_32k={str(report['q2_ramp_gate']['failed_32k_q2']).lower()}")
    print()
    print("## Archive checksums")
    for row in report["archives"]:
        print(
            f"- {row['name']}: ok={str(row['ok']).lower()} | "
            f"files={row['files']} | {row['sha256sums']}"
        )
    print()
    print("## Incomplete items")
    for row in report["incomplete_items"]:
        print(f"- {row['item']}: {row['status']} | {row['note']}")
    print()
    print("## Guardrails")
    for row in report["guardrails"]:
        print(f"- {row['item']}: {row['status']} | {row['note']}")
    print()
    print("## Q2 ramp gate")
    print(report["q2_ramp_gate"]["command_hint"])
    for step in report["q2_ramp_gate"]["steps"]:
        print(f"- {step['prompt']}: any_ok={str(step['any_ok']).lower()}")
    print()
    print("## Next dry-run commands")
    for command in report["dry_run_commands"]:
        print(command)
    print()
    print("## Recommended next actions")
    for action in report["recommended_next_actions"]:
        print(f"- {action}")
    print()
    print("## Real command printing")
    for command in report["real_command_printing"]:
        print(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report current no-GPU readiness for cautious 32k KV testing.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-gpu", action="store_true", help="skip even the read-only nvidia-smi snapshot")
    args = parser.parse_args()

    report = build_report(no_gpu=args.no_gpu)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
