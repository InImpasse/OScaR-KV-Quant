#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from pathlib import Path

import report_q2_ramp_gate


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts/bench_32k_llamacpp_kv.sh"
TIMEOUT_BY_PROMPT = {
    512: 45,
    2048: 60,
    4096: 75,
    8192: 90,
    16384: 240,
    32768: 480,
}
SUFFIX_BY_PROMPT = {
    512: "512",
    2048: "2k",
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
}


def command_env(args: argparse.Namespace, gate: dict) -> dict[str, str]:
    prompt = gate["next_prompt"]
    if prompt is None:
        raise SystemExit("q2 ramp is complete; no next prompt to run")
    if prompt == 32768 and gate["recommendation"] == "hold_32k_q2" and not args.ack_q2_ramp_gate_hold:
        raise SystemExit("q2 ramp gate is hold_32k_q2; refusing 32k without --ack-q2-ramp-gate-hold")

    env = {
        "OUT_DIR": f"{args.out_prefix}_{SUFFIX_BY_PROMPT[prompt]}_{args.case}",
        "CASES": args.case,
        "GEN_TOKENS": "1",
        "REPETITIONS": "1",
        "VRAM_POLL_INTERVAL": str(args.vram_poll_interval),
        "MAX_PEAK_MIB": str(args.max_peak_mib),
        "POST_CASE_COOLDOWN_SEC": str(args.post_case_cooldown_sec),
        "DRY_RUN": "0" if args.real else "1",
        "PROMPT_TOKENS": str(prompt),
        "CASE_TIMEOUT_SEC": str(TIMEOUT_BY_PROMPT[prompt]),
    }
    if prompt >= 32768:
        env.update({
            "ACK_HEAVY_32K": "1",
            "ACK_Q2_32K_NOGO": "1",
            "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0",
        })
    return env


def build_command(env: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in env.items()] + [str(BENCH)]


def run_command(env: dict[str, str]) -> int:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run([str(BENCH)], cwd=ROOT, env=merged, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print or run only the next q2/int2 recovery-ramp step from the archived ramp gate."
    )
    parser.add_argument("--case", choices=("plain_int2", "oscar_int2"), default="plain_int2")
    parser.add_argument("--real", action="store_true", help="execute the next step instead of printing a dry-run command")
    parser.add_argument("--ack-real", action="store_true", help="required with --real")
    parser.add_argument("--ack-q2-ramp-gate-hold", action="store_true", help="required if the next step is held 32k q2")
    parser.add_argument("--max-peak-mib", type=int, default=7000)
    parser.add_argument("--post-case-cooldown-sec", type=int, default=30)
    parser.add_argument("--vram-poll-interval", type=float, default=0.5)
    parser.add_argument("--out-prefix", default="/tmp/llamacpp_q2_ramp_next")
    args = parser.parse_args()

    if args.real and not args.ack_real:
        raise SystemExit("--real executes the next ramp step and requires --ack-real")
    if args.max_peak_mib < 0:
        raise SystemExit("--max-peak-mib must be non-negative")
    if args.post_case_cooldown_sec < 0:
        raise SystemExit("--post-case-cooldown-sec must be non-negative")
    if args.vram_poll_interval <= 0:
        raise SystemExit("--vram-poll-interval must be positive")

    gate = report_q2_ramp_gate.build_report(report_q2_ramp_gate.DEFAULT_MATRIX)
    env = command_env(args, gate)
    command = build_command(env)
    print(
        f"# q2 ramp next: prompt={env['PROMPT_TOKENS']} case={args.case} "
        f"recommendation={gate['recommendation']}"
    )
    print(shlex.join(command))

    if not args.real:
        print("# Dry run only; add --real --ack-real to execute this single next step.")
        return

    raise SystemExit(run_command(env))


if __name__ == "__main__":
    main()
