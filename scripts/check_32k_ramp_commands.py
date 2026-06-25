#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/print_32k_q2_ramp_commands.py"


def run(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(HELPER), *args], text=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def command_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def line_for_prompt(text: str, prompt: str) -> str:
    matches = [line for line in command_lines(text) if f"PROMPT_TOKENS={prompt}" in line]
    require(len(matches) == 1, f"expected one command for prompt {prompt}, got {len(matches)}")
    return matches[0]


def require_no_output_dirs(prefix: str) -> None:
    matches = list(Path("/tmp").glob(f"{Path(prefix).name}*"))
    require(not matches, "command printer should not create output directories: " + ", ".join(str(p) for p in matches))


def main() -> None:
    out_prefix = "/tmp/llamacpp_q2_ramp_checker_no_side_effect"
    dry = run("--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    require("DRY_RUN=1" in dry, "default ramp commands must be dry-run")
    require("PROMPT_TOKENS=512" in dry, "recovery ramp must include 512 smoke")
    require("PROMPT_TOKENS=2048" in dry, "recovery ramp must include 2k probe")
    require("PROMPT_TOKENS=4096" in dry, "recovery ramp must include 4k probe")
    require("PROMPT_TOKENS=8192" in dry, "ramp must include 8k")
    require("PROMPT_TOKENS=16384" in dry, "ramp must include 16k")
    require("PROMPT_TOKENS=32768" in dry, "ramp must include 32k")
    require("CASE_TIMEOUT_SEC=240" in line_for_prompt(dry, "16384"), "16k ramp should allow known-completing timeout")
    require("ACK_HEAVY_32K=1" in dry, "32k q2 command must include heavy ACK")
    require("ACK_Q2_32K_NOGO=1" in dry, "32k q2 command must include NO-GO ACK")
    require("MAX_PEAK_MIB=7000" in dry, "ramp must include peak-VRAM watchdog")
    require("POST_CASE_COOLDOWN_SEC=30" in dry, "ramp must include post-case cooldown")
    require("--real --ack-real" in dry, "dry-run guidance should require --ack-real")
    require("--ack-32k-q2-real" in HELPER.read_text(), "ramp helper should require separate 32k q2 real ACK")
    require("--ack-q2-ramp-gate-hold" in HELPER.read_text(), "ramp helper should require separate ramp gate hold ACK")
    require("ACK_Q2_RAMP_GATE_HOLD=0" in dry, "32k q2 dry-run command should expose ramp gate hold ACK default")
    require("subprocess" not in HELPER.read_text(), "ramp helper must not execute commands")

    missing_ack = subprocess.run([sys.executable, str(HELPER), "--real", "--out-prefix", out_prefix], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require_no_output_dirs(out_prefix)
    require(missing_ack.returncode != 0, "--real without --ack-real should fail")
    require("--ack-real" in missing_ack.stderr, "--real failure should mention --ack-real")

    real = run("--real", "--ack-real", "--case", "oscar_int2", "--max-peak-mib", "6500", "--post-case-cooldown-sec", "45", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    require("DRY_RUN=0" in line_for_prompt(real, "512"), "512 should print real command with --real --ack-real")
    require("DRY_RUN=0" in line_for_prompt(real, "2048"), "2k should print real command with --real --ack-real")
    require("DRY_RUN=0" in line_for_prompt(real, "4096"), "4k should print real command with --real --ack-real")
    require("DRY_RUN=0" in line_for_prompt(real, "8192"), "8k should print real command with --real --ack-real")
    require("DRY_RUN=0" in line_for_prompt(real, "16384"), "16k should print real command with --real --ack-real")
    require("DRY_RUN=1" in line_for_prompt(real, "32768"), "32k should stay dry-run without q2 real ACK")
    require("--ack-32k-q2-real" in real, "real ramp output should explain q2 real ACK")
    require("CASES=oscar_int2" in real, "--case should select oscar_int2")
    require("MAX_PEAK_MIB=6500" in real, "--max-peak-mib should be propagated")
    require("POST_CASE_COOLDOWN_SEC=45" in real, "--post-case-cooldown-sec should be propagated")

    q2_real = run("--real", "--ack-real", "--ack-32k-q2-real", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    require("DRY_RUN=1" in line_for_prompt(q2_real, "32768"), "32k should remain dry-run without ramp gate hold ACK")
    require("--ack-q2-ramp-gate-hold" in q2_real, "q2-ACK ramp output should explain ramp gate hold ACK")

    q2_hold_real = run("--real", "--ack-real", "--ack-32k-q2-real", "--ack-q2-ramp-gate-hold", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    require("DRY_RUN=0" in line_for_prompt(q2_hold_real, "32768"), "32k should print real command only with q2 and ramp gate ACKs")
    require("ACK_Q2_RAMP_GATE_HOLD=1" in line_for_prompt(q2_hold_real, "32768"), "32k real command should carry ramp gate hold ACK")

    print("32k q2 ramp command checks passed")


if __name__ == "__main__":
    main()
