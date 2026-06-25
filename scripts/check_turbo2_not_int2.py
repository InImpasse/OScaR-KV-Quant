#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    audit = (ROOT / "scripts/audit_goal_status.py").read_text()
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()

    require("32k_turbo2_reference" in audit, "audit must track Turbo2 as a separate reference")
    require("Turbo2 is not exact OSCAR INT2" in audit, "audit must state Turbo2 is not exact INT2")
    require("oscar_int2_32k" in audit, "audit must inspect exact oscar_int2 32k row")
    require("q2_0/q2_0" in audit, "audit must require exact q2_0/q2_0 for INT2")

    int2_block = audit.split('"32k_int2_speed_target"', 1)[1].split("graph_no_help", 1)[0]
    require("oscar_turbo2_streamk_32k is not None" not in int2_block,
            "32k INT2 target must not be satisfied by oscar_turbo2_streamk")
    require('exact_q2_kv=' in int2_block and 'q2_0/q2_0' in audit,
            "32k INT2 target must require exact q2_0/q2_0 KV")

    require("`oscar_turbo2_streamk` is not `oscar_int2`" in report,
            "triage report must explicitly separate oscar_turbo2_streamk and oscar_int2")
    require("Reference int2 format boundary" in report and "asymmetric scale/zero" in report,
            "triage report must document why reference int2 is not llama.cpp q2_0")
    require("Do not redo OSCAR rotation yet" in report,
            "triage report must record the current rotation decision")

    print("turbo2/int2 separation checks passed")


if __name__ == "__main__":
    main()
