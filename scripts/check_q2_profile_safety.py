#!/usr/bin/env python3
import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def index(text: str, needle: str) -> int:
    pos = text.find(needle)
    if pos < 0:
        raise AssertionError(f"missing expected text: {needle}")
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Check q2 profiling helpers stay low-risk by default.")
    parser.add_argument("--profile", type=Path, default=Path("scripts/q2_profile.sh"))
    parser.add_argument("--segments", type=Path, default=Path("scripts/q2_segment_bench.sh"))
    args = parser.parse_args()

    profile = args.profile.read_text()
    segments = args.segments.read_text()

    for name, text in (("q2_profile.sh", profile), ("q2_segment_bench.sh", segments)):
        require('build-cuda/bin/llama-bench' in text, f"{name} must default to build-cuda llama-bench")
        require('DRY_RUN="${DRY_RUN:-1}"' in text, f"{name} must default to DRY_RUN=1")
        require('Set DRY_RUN=0' in text, f"{name} must tell users how to opt into real execution")

    require(
        index(profile, 'if [[ "$DRY_RUN" == "1" ]]; then') < index(profile, 'scripts/ncu_wsl_preflight.sh'),
        "q2_profile dry-run must exit before profiler preflight",
    )
    require(
        'Q2_PROFILE_GPU_SNAPSHOT="${Q2_PROFILE_GPU_SNAPSHOT:-0}"' in profile,
        "q2_profile must skip nvidia-smi by default",
    )
    require(
        "Q2_PROFILE_GPU_SNAPSHOT=1" in profile,
        "q2_profile must document GPU snapshot opt-in",
    )
    require(
        'OUT_DIR="$OUT/segments" "$ROOT_DIR/scripts/q2_segment_bench.sh"' in profile,
        "q2_profile fallback must pass OUT_DIR as an environment variable",
    )
    require(
        index(segments, 'if [[ "$DRY_RUN" == "1" ]]; then') < index(segments, 'bench_one q2q2_pp8192'),
        "q2_segment_bench dry-run must exit before benchmark cases",
    )

    print("q2 profiling helper safety checks passed")


if __name__ == "__main__":
    main()
