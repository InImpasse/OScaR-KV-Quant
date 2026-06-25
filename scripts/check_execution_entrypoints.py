#!/usr/bin/env python3
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SKIP_DIR_NAMES = {
    "__pycache__",
}

RISK_MARKERS = (
    "llama-cli",
    "llama-bench",
    "llama-perplexity",
    "measure_vram.sh",
    "ncu",
    "nsys",
)

SAFE_HELPERS = {
    "scripts/audit_goal_status.py",
    "scripts/check_build_defaults.py",
    "scripts/check_32k_ramp_commands.py",
    "scripts/check_32k_matrix_commands.py",
    "scripts/check_execution_entrypoints.py",
    "scripts/check_futuremls_q2_cuda_plan.py",
    "scripts/check_kv_bench_gate.py",
    "scripts/check_legacy_bench_safety.py",
    "scripts/check_no_gpu_verifier_no_nvidia_smi.py",
    "scripts/check_post_case_cooldown.py",
    "scripts/check_q2_profile_safety.py",
    "scripts/check_q2_ramp_gate.py",
    "scripts/check_q2_ramp_gate_harness_guard.py",
    "scripts/check_recovery_readiness.py",
    "scripts/check_watchdog_summary_fields.py",
    "scripts/check_mixed_vec_int2_ramp.py",
    "scripts/check_mixed_vec_quality_expansion.py",
    "scripts/check_mixed_vec_smoke.py",
    "scripts/check_mixed_vec_tiled_decision.py",
    "scripts/print_mixed_vec_int2_ramp_commands.py",
    "scripts/print_32k_matrix_commands.py",
    "scripts/refresh_run_archive_checksums.py",
    "scripts/report_recovery_readiness.py",
    "scripts/report_q2_ramp_gate.py",
    "scripts/print_32k_q2_ramp_commands.py",
    "scripts/summarize_mixed_vec_int2_ramp.py",
    "scripts/summarize_mixed_vec_smoke.py",
    "scripts/summarize_32k_llamacpp_kv.py",
    "scripts/summarize_cuda_graph_ab.py",
    "scripts/summarize_llamacpp_accuracy_suite.py",
    "scripts/summarize_llamacpp_matrix.py",
    "scripts/summarize_kv_matrix.py",
    "scripts/summarize_kv_ppl.py",
}

INTENTIONAL_TOOLS = {
    "scripts/build_llamacpp.sh": ("cmake --build",),
    "scripts/check_kv_env.sh": ("--help", "--list-devices"),
    "scripts/measure_vram.sh": ("nvidia-smi",),
    "scripts/ncu_wsl_preflight.sh": ("ncu WSL preflight",),
}

DRY_RUN_ENTRYPOINTS = {
    "scripts/bench_32k_llamacpp_kv.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "Dry run complete; no results written."),
    "scripts/bench_llamacpp_matrix.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "dry run complete"),
    "scripts/bench_kv_cache.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_HEAVY_CONTEXT"),
    "scripts/bench_kv_cache_matrix.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_MATRIX_BENCH"),
    "scripts/cuda_graph_ab.sh": ("RUN_REAL=\"${RUN_REAL:-0}\"", "Dry run complete; set RUN_REAL=1"),
    "scripts/cuda_graph_compare_llamacpp_matrix.sh": ("RUN_REAL=\"${RUN_REAL:-0}\"", "dry run complete"),
    "scripts/q2_profile.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "Set DRY_RUN=0"),
    "scripts/q2_segment_bench.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "Set DRY_RUN=0"),
    "scripts/run_kv_ppl_matrix.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_PPL_MATRIX"),
    "scripts/run_llamacpp_accuracy_suite.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_EVAL"),
    "scripts/run_llamacpp_lcb_v6.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ALLOW_CODE_EXEC"),
    "scripts/run_mixed_vec_int2_ramp.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_MIXED_VEC_RAMP"),
    "scripts/run_mixed_vec_quality_expansion.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_QUALITY_EXPANSION"),
    "scripts/run_mixed_vec_smoke.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_MIXED_VEC_SMOKE"),
    "scripts/run_q2_tile_ab.sh": ("RUN_REAL=\"${RUN_REAL:-0}\"", "Dry run: set RUN_REAL=1"),
    "scripts/run_llamacpp.sh": ("DRY_RUN=\"${DRY_RUN:-1}\"", "ACK_RUN_LLAMA"),
    "scripts/run_gpqa_gsm8k_cli_eval.py": ("--real --ack-eval", "Dry run complete"),
    "scripts/run_q2_ramp_next.py": ("--real executes the next ramp step", "--ack-real"),
    "scripts/verify_llamacpp_32k_kv_no_gpu.sh": (
        "CHECK_GPU_SNAPSHOT=\"${CHECK_GPU_SNAPSHOT:-0}\"",
        "set CHECK_GPU_SNAPSHOT=1",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def risky(text: str) -> bool:
    return any(marker in text for marker in RISK_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check scripts that can launch GPU/profiler workloads are guarded.")
    parser.parse_args()

    failures = []
    for path in sorted(SCRIPTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        rel_name = rel.as_posix()
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        text = path.read_text(errors="ignore")
        if not risky(text):
            continue
        if rel_name in SAFE_HELPERS:
            continue
        if rel_name in INTENTIONAL_TOOLS:
            for needle in INTENTIONAL_TOOLS[rel_name]:
                if needle not in text:
                    failures.append(f"{rel_name}: missing expected intentional-tool marker: {needle}")
            continue
        if rel_name in DRY_RUN_ENTRYPOINTS:
            for needle in DRY_RUN_ENTRYPOINTS[rel_name]:
                if needle not in text:
                    failures.append(f"{rel_name}: missing dry-run/ACK marker: {needle}")
            if "DRY_RUN=\"${DRY_RUN:-0}\"" in text or "RUN_REAL=\"${RUN_REAL:-1}\"" in text:
                failures.append(f"{rel_name}: defaults to real execution")
            continue
        failures.append(f"{rel_name}: risky execution marker without explicit safety classification")

    require(not failures, "unsafe execution entrypoints:\n" + "\n".join(failures))
    print("execution entrypoint safety checks passed")


if __name__ == "__main__":
    main()
