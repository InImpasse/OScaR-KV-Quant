#!/usr/bin/env python3
import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_mixed_vec_int2_ramp.sh"


def emit(title: str, env: dict[str, str]) -> None:
    prefix = [f"{key}={value}" for key, value in env.items()]
    print(f"# {title}")
    print(shlex.join(prefix + [str(RUNNER)]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print mixed vec INT2 ramp commands. Never executes benchmarks."
    )
    parser.add_argument("--variant", choices=("plain_int2", "oscar_int2"), default="oscar_int2")
    parser.add_argument("--mode", choices=("all", "pure", "mixed_fused", "mixed_vec"), default="all")
    parser.add_argument("--include-6144", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--ack-real", action="store_true")
    parser.add_argument("--max-peak-mib", type=int, default=7000)
    parser.add_argument("--post-case-cooldown-sec", type=int, default=30)
    parser.add_argument("--out-prefix", default="/tmp/mixed_vec_int2_ramp")
    args = parser.parse_args()

    if args.real and not args.ack_real:
        raise SystemExit("--real requires --ack-real")

    common = {
        "VARIANT": args.variant,
        "MODE": args.mode,
        "INCLUDE_6144": "1" if args.include_6144 else "0",
        "GEN_TOKENS": "1",
        "REPETITIONS": "1",
        "MAX_PEAK_MIB": str(args.max_peak_mib),
        "POST_CASE_COOLDOWN_SEC": str(args.post_case_cooldown_sec),
        "DRY_RUN": "0" if args.real else "1",
        "ACK_MIXED_VEC_RAMP": "1" if args.real else "0",
    }

    emit(
        f"full ramp ({args.mode}/{args.variant})",
        {"OUT_DIR": f"{args.out_prefix}_full", **common},
    )

    for title, prompt in (
        ("512 smoke", "512"),
        ("2k probe", "2048"),
        ("4k official ramp", "4096"),
        ("6144 optional probe", "6144"),
        ("8k gate", "8192"),
    ):
        if prompt == "6144" and not args.include_6144:
            continue
        emit(
            f"{title} single-step ({args.mode}/{args.variant})",
            {
                "OUT_DIR": f"{args.out_prefix}_{prompt}_{args.mode}_{args.variant}",
                "PROMPT_TOKENS": prompt,
                **common,
            },
        )

    if not args.real:
        print("# Real commands require --real --ack-real")


if __name__ == "__main__":
    main()
