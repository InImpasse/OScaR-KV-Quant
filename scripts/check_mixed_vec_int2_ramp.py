#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mixed_vec_int2_ramp.sh"
PRINTER = ROOT / "scripts/print_mixed_vec_int2_ramp_commands.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def main() -> None:
    text = RUNNER.read_text()
    require('DRY_RUN="${DRY_RUN:-1}"' in text, "ramp runner must default to DRY_RUN=1")
    require("ACK_MIXED_VEC_RAMP" in text, "ramp runner must require ACK_MIXED_VEC_RAMP")
    require("512" in text and "8192" in text, "ramp runner must include 512 and 8192 steps")
    require("6144" in text, "ramp runner must support optional 6144 probe")
    require("mixed_vec" in text and "mixed_fused" in text and "pure" in text,
            "ramp runner must compare pure, mixed_fused, and mixed_vec modes")

    dry = run(["bash", str(RUNNER)])
    require(dry.returncode == 0, f"default dry-run failed: {dry.stderr}")
    require("Dry run complete" in dry.stdout, "ramp dry-run should finish cleanly")
    require("mixed_vec_p8192" in dry.stdout or "mixed_vec_p8192_oscar_int2" in dry.stdout,
            "dry-run should include 8k mixed_vec case")

    printer = run([sys.executable, str(PRINTER)])
    require(printer.returncode == 0, f"ramp printer failed: {printer.stderr}")
    require("PROMPT_TOKENS=512" in printer.stdout, "printer must include 512")
    require("PROMPT_TOKENS=8192" in printer.stdout, "printer must include 8192")
    require("subprocess" not in PRINTER.read_text(), "ramp printer must not execute commands")

    missing_ack = run(["bash", str(RUNNER), "-c", "DRY_RUN=0 bash scripts/run_mixed_vec_int2_ramp.sh"])
    # direct env test via subprocess with env
    missing_ack = subprocess.run(
        ["bash", "-lc", "DRY_RUN=0 scripts/run_mixed_vec_int2_ramp.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(missing_ack.returncode != 0, "real ramp without ACK should fail")
    require("ACK_MIXED_VEC_RAMP" in missing_ack.stderr, "real refusal should mention ACK")

    print("mixed vec INT2 ramp checks passed")


if __name__ == "__main__":
    main()
