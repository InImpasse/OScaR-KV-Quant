#!/usr/bin/env python3
import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def emit(title: str, env: dict[str, str]) -> None:
    command = [str(ROOT / "scripts/bench_32k_llamacpp_kv.sh")]
    prefix = [f"{key}={value}" for key, value in env.items()]
    print(f"# {title}")
    print(shlex.join(prefix + command))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a cautious llama.cpp q2/int2 recovery ramp. This helper never executes benchmark commands."
    )
    parser.add_argument("--case", choices=("plain_int2", "oscar_int2"), default="plain_int2")
    parser.add_argument("--real", action="store_true", help="print DRY_RUN=0 commands; still does not execute them")
    parser.add_argument("--ack-real", action="store_true", help="required with --real to print DRY_RUN=0 commands")
    parser.add_argument("--ack-32k-q2-real", action="store_true", help="also print the 32k q2/int2 command with DRY_RUN=0")
    parser.add_argument("--ack-q2-ramp-gate-hold", action="store_true", help="also acknowledge the current hold_32k_q2 ramp gate")
    parser.add_argument("--max-peak-mib", type=int, default=7000)
    parser.add_argument("--post-case-cooldown-sec", type=int, default=30)
    parser.add_argument("--out-prefix", default="/tmp/llamacpp_q2_ramp")
    args = parser.parse_args()

    if args.max_peak_mib < 0:
        raise SystemExit("--max-peak-mib must be non-negative")
    if args.post_case_cooldown_sec < 0:
        raise SystemExit("--post-case-cooldown-sec must be non-negative")
    if args.real and not args.ack_real:
        raise SystemExit("--real only prints commands, but still requires --ack-real")

    common = {
        "CASES": args.case,
        "GEN_TOKENS": "1",
        "REPETITIONS": "1",
        "VRAM_POLL_INTERVAL": "0.5",
        "MAX_PEAK_MIB": str(args.max_peak_mib),
        "POST_CASE_COOLDOWN_SEC": str(args.post_case_cooldown_sec),
    }

    for title, suffix, prompt, timeout in (
        ("512 q2/int2 post-crash smoke", "512", "512", "45"),
        ("2k q2/int2 low-load probe", "2k", "2048", "60"),
        ("4k q2/int2 low-load probe", "4k", "4096", "75"),
        ("8k q2/int2 sanity", "8k", "8192", "90"),
        ("16k q2/int2 gate", "16k", "16384", "240"),
    ):
        emit(
            title,
            {
                "OUT_DIR": f"{args.out_prefix}_{suffix}_{args.case}",
                **common,
                "DRY_RUN": "0" if args.real else "1",
                "PROMPT_TOKENS": prompt,
                "CASE_TIMEOUT_SEC": timeout,
            },
        )

    emit(
        "32k q2/int2 single-case attempt after every smaller step is healthy",
        {
            "OUT_DIR": f"{args.out_prefix}_32k_{args.case}",
            **common,
            "DRY_RUN": "0" if args.real and args.ack_32k_q2_real and args.ack_q2_ramp_gate_hold else "1",
            "PROMPT_TOKENS": "32768",
            "CASE_TIMEOUT_SEC": "480",
            "ACK_HEAVY_32K": "1",
            "ACK_Q2_32K_NOGO": "1",
            "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0",
        },
    )

    if not args.real:
        print("# Commands above are dry-runs. Re-run this helper with --real --ack-real to print DRY_RUN=0 commands.")
    elif not args.ack_32k_q2_real:
        print("# 32k q2/int2 command remains DRY_RUN=1. Add --ack-32k-q2-real to print it as DRY_RUN=0.")
    elif not args.ack_q2_ramp_gate_hold:
        print("# 32k q2/int2 command remains DRY_RUN=1 while ramp gate is hold_32k_q2. Add --ack-q2-ramp-gate-hold only for a deliberate post-change validation.")


if __name__ == "__main__":
    main()
