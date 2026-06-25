#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_q2_ramp_next.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_no_output_dirs(prefix: str) -> None:
    matches = list(Path("/tmp").glob(f"{Path(prefix).name}*"))
    require(not matches, "q2 next runner should not create output dirs in dry-run/checker paths: " + ", ".join(str(p) for p in matches))


def main() -> None:
    prefix = "/tmp/llamacpp_q2_ramp_next_checker_no_side_effect"
    dry = run("--out-prefix", prefix, "--case", "oscar_int2")
    require(dry.returncode == 0, f"default dry-run should succeed: {dry.stderr}")
    require_no_output_dirs(prefix)
    require("q2 ramp next: prompt=512" in dry.stdout, "runner should choose current next prompt 512")
    require("recommendation=hold_32k_q2" in dry.stdout, "runner should expose current hold recommendation")
    require("PROMPT_TOKENS=512" in dry.stdout, "runner command should use 512 prompt")
    require("CASES=oscar_int2" in dry.stdout, "runner should honor selected case")
    require("DRY_RUN=1" in dry.stdout, "runner should default to dry-run")
    require("MAX_PEAK_MIB=7000" in dry.stdout, "runner should carry peak watchdog")
    require("POST_CASE_COOLDOWN_SEC=30" in dry.stdout, "runner should carry cooldown")
    require("ACK_Q2_RAMP_GATE_HOLD" not in dry.stdout, "512 next step should not carry 32k hold ACK")

    missing_ack = run("--real", "--out-prefix", prefix)
    require(missing_ack.returncode != 0, "--real without --ack-real should fail")
    require("--ack-real" in missing_ack.stderr, "real refusal should mention --ack-real")
    require_no_output_dirs(prefix)

    text = RUNNER.read_text()
    require("shell=True" not in text, "runner must not use shell=True")
    require("report_q2_ramp_gate.build_report" in text, "runner must derive next step from ramp gate")
    require("run_command(env)" in text, "runner should execute only after explicit real ACK")

    print("q2 ramp next runner checks passed")


if __name__ == "__main__":
    main()
