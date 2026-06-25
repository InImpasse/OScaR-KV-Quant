#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/print_32k_matrix_commands.py"


def run(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(HELPER), *args], text=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def command_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def line_for(text: str, case: str) -> str:
    matches = [line for line in command_lines(text) if f"CASES={case}" in line]
    require(len(matches) == 1, f"expected one command for {case}, got {len(matches)}")
    return matches[0]


def require_no_output_dirs(prefix: str) -> None:
    matches = list(Path("/tmp").glob(f"{Path(prefix).name}*"))
    require(not matches, "command printer should not create output directories: " + ", ".join(str(p) for p in matches))


def main() -> None:
    out_prefix = "/tmp/llamacpp_32k_matrix_checker_no_side_effect"
    dry = run("--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    lines = command_lines(dry)
    require(len(lines) == 11, f"expected eleven benchmark commands, got {len(lines)}")
    require("DRY_RUN=1" in dry, "matrix helper must default to dry-run commands")
    for case in ("baseline_bf16", "oscar_turbo2_streamk", "turbo2_streamk", "oscar_int4", "plain_int4", "oscar_turbo3", "plain_int3", "oscar_kq4_vq2", "oscar_kq4_vturbo3", "plain_int2", "oscar_int2"):
        require(f"CASES={case}" in dry, f"missing case {case}")
    require("plain_int3 maps to llama.cpp TurboQuant KV cache turbo3/turbo3" in dry,
            "plain_int3 should be documented as TurboQuant 3-bit KV")
    require("oscar_kq4_vq2 is a mixed q4_0/q2_0 quality variant, not exact OSCAR INT2" in dry,
            "oscar_kq4_vq2 should be documented as a mixed quality variant")
    require("oscar_kq4_vturbo3 is a mixed q4_0/turbo3 quality variant, not exact OSCAR INT2" in dry,
            "oscar_kq4_vturbo3 should be documented as a mixed quality variant")
    require(dry.count("ACK_HEAVY_32K=1") == 4, "only 2-bit 32k commands should carry heavy ACK")
    require(dry.count("ACK_Q2_32K_NOGO=1") == 4, "only 2-bit 32k commands should carry NO-GO ACK")
    require("MAX_PEAK_MIB=7000" in dry, "matrix commands should include peak-VRAM watchdog")
    require("POST_CASE_COOLDOWN_SEC=30" in dry, "matrix commands should include post-case cooldown")
    require("--real --ack-real" in dry, "dry-run guidance should require --ack-real")
    require("--ack-32k-q2-real" in HELPER.read_text(), "matrix helper should require separate q2 real ACK")
    require("--ack-q2-ramp-gate-hold" in HELPER.read_text(), "matrix helper should require separate ramp gate hold ACK")
    require(dry.count("ACK_Q2_RAMP_GATE_HOLD=0") == 4, "2-bit 32k commands should expose ramp gate hold ACK default")
    require("subprocess" not in HELPER.read_text(), "matrix helper must not execute commands")

    missing_ack = subprocess.run([sys.executable, str(HELPER), "--real", "--out-prefix", out_prefix], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require_no_output_dirs(out_prefix)
    require(missing_ack.returncode != 0, "--real without --ack-real should fail")
    require("--ack-real" in missing_ack.stderr, "--real failure should mention --ack-real")

    real = run("--real", "--ack-real", "--max-peak-mib", "6500", "--post-case-cooldown-sec", "45", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    for case in ("baseline_bf16", "oscar_int4", "plain_int4", "oscar_turbo3", "plain_int3", "oscar_kq4_vq2", "oscar_kq4_vturbo3"):
        require("DRY_RUN=0" in line_for(real, case), f"{case} should print real command with --real --ack-real")
    for case in ("oscar_turbo2_streamk", "turbo2_streamk", "plain_int2", "oscar_int2"):
        require("DRY_RUN=1" in line_for(real, case), f"{case} should stay dry-run without q2 real ACK")
    require("--ack-32k-q2-real" in real, "real matrix output should explain q2 real ACK")
    require("MAX_PEAK_MIB=6500" in real, "--max-peak-mib should be propagated")
    require("POST_CASE_COOLDOWN_SEC=45" in real, "--post-case-cooldown-sec should be propagated")

    q2_real = run("--real", "--ack-real", "--ack-32k-q2-real", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    for case in ("oscar_turbo2_streamk", "turbo2_streamk", "plain_int2", "oscar_int2"):
        require("DRY_RUN=1" in line_for(q2_real, case), f"{case} should remain dry-run without ramp gate hold ACK")
    require("--ack-q2-ramp-gate-hold" in q2_real, "q2-ACK matrix output should explain ramp gate hold ACK")

    q2_hold_real = run("--real", "--ack-real", "--ack-32k-q2-real", "--ack-q2-ramp-gate-hold", "--out-prefix", out_prefix)
    require_no_output_dirs(out_prefix)
    for case in ("oscar_turbo2_streamk", "turbo2_streamk", "plain_int2", "oscar_int2"):
        require("DRY_RUN=0" in line_for(q2_hold_real, case), f"{case} should print real command only with q2 and ramp gate ACKs")
        require("ACK_Q2_RAMP_GATE_HOLD=1" in line_for(q2_hold_real, case), f"{case} real command should carry ramp gate hold ACK")

    print("32k matrix command checks passed")


if __name__ == "__main__":
    main()
