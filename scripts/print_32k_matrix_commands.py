#!/usr/bin/env python3
import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts/bench_32k_llamacpp_kv.sh"


def emit(title: str, env: dict[str, str]) -> None:
    print(f"# {title}")
    print(shlex.join([f"{key}={value}" for key, value in env.items()] + [str(BENCH)]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print llama.cpp-only 32k KV matrix commands. This helper never executes benchmarks."
    )
    parser.add_argument("--real", action="store_true", help="print DRY_RUN=0 commands; still does not execute them")
    parser.add_argument("--ack-real", action="store_true", help="required with --real to print DRY_RUN=0 commands")
    parser.add_argument("--ack-32k-q2-real", action="store_true", help="also print 32k q2/int2 commands with DRY_RUN=0")
    parser.add_argument("--ack-q2-ramp-gate-hold", action="store_true", help="also acknowledge the current hold_32k_q2 ramp gate")
    parser.add_argument("--max-peak-mib", type=int, default=7000)
    parser.add_argument("--post-case-cooldown-sec", type=int, default=30)
    parser.add_argument("--out-prefix", default="/tmp/llamacpp_32k_matrix")
    args = parser.parse_args()

    if args.max_peak_mib < 0:
        raise SystemExit("--max-peak-mib must be non-negative")
    if args.post_case_cooldown_sec < 0:
        raise SystemExit("--post-case-cooldown-sec must be non-negative")
    if args.real and not args.ack_real:
        raise SystemExit("--real only prints commands, but still requires --ack-real")

    common = {
        "PROMPT_TOKENS": "32768",
        "GEN_TOKENS": "1",
        "REPETITIONS": "1",
        "VRAM_POLL_INTERVAL": "0.5",
        "MAX_PEAK_MIB": str(args.max_peak_mib),
        "POST_CASE_COOLDOWN_SEC": str(args.post_case_cooldown_sec),
    }
    cases = [
        ("baseline_bf16", "90", {}),
        ("oscar_turbo2_streamk", "180", {"ACK_HEAVY_32K": "1", "ACK_Q2_32K_NOGO": "1", "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0"}),
        ("turbo2_streamk", "180", {"ACK_HEAVY_32K": "1", "ACK_Q2_32K_NOGO": "1", "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0"}),
        ("oscar_int4", "120", {}),
        ("plain_int4", "120", {}),
        ("oscar_turbo3", "180", {}),
        ("plain_int3", "180", {}),
        ("oscar_kq4_vq2", "180", {}),
        ("oscar_kq4_vturbo3", "240", {}),
        ("plain_int2", "480", {"ACK_HEAVY_32K": "1", "ACK_Q2_32K_NOGO": "1", "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0"}),
        ("oscar_int2", "480", {"ACK_HEAVY_32K": "1", "ACK_Q2_32K_NOGO": "1", "ACK_Q2_RAMP_GATE_HOLD": "1" if args.ack_q2_ramp_gate_hold else "0"}),
    ]

    for case, timeout, extra in cases:
        is_q2 = case in ("oscar_turbo2_streamk", "turbo2_streamk", "plain_int2", "oscar_int2")
        dry_run = "0" if args.real and (not is_q2 or (args.ack_32k_q2_real and args.ack_q2_ramp_gate_hold)) else "1"
        emit(
            f"32k {case}",
            {
                "OUT_DIR": f"{args.out_prefix}_{case}",
                "CASES": case,
                **common,
                "DRY_RUN": dry_run,
                "CASE_TIMEOUT_SEC": timeout,
                **extra,
            },
        )

    print("# plain_int3 maps to llama.cpp TurboQuant KV cache turbo3/turbo3; Q3_K remains a weight format, not a KV cache type.")
    print("# oscar_kq4_vq2 is a mixed q4_0/q2_0 quality variant, not exact OSCAR INT2.")
    print("# oscar_kq4_vturbo3 is a mixed q4_0/turbo3 quality variant, not exact OSCAR INT2.")
    if not args.real:
        print("# Commands above are dry-runs. Re-run this helper with --real --ack-real to print DRY_RUN=0 commands.")
    elif not args.ack_32k_q2_real:
        print("# 32k q2/int2 commands remain DRY_RUN=1. Add --ack-32k-q2-real to print them as DRY_RUN=0.")
    elif not args.ack_q2_ramp_gate_hold:
        print("# 32k q2/int2 commands remain DRY_RUN=1 while ramp gate is hold_32k_q2. Add --ack-q2-ramp-gate-hold only for deliberate post-change validation.")


if __name__ == "__main__":
    main()
