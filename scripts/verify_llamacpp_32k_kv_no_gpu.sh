#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECK_GPU_SNAPSHOT="${CHECK_GPU_SNAPSHOT:-0}"
MATRIX_DIR="runs/llamacpp_32k_kv_matrix_current"
COMBINED_CSV="$MATRIX_DIR/combined.csv"
SHA256SUMS="$MATRIX_DIR/SHA256SUMS"
GRAPH_AB_DIR="runs/cuda_graph_ab_512_current"
GRAPH_AB_SHA256SUMS="$GRAPH_AB_DIR/SHA256SUMS"
GOAL_STATUS_DIR="runs/goal_status_current"
GOAL_STATUS_SHA256SUMS="$GOAL_STATUS_DIR/SHA256SUMS"
Q2_CUDA_PATH_DIR="runs/q2_cuda_path_current"
Q2_CUDA_PATH_SHA256SUMS="$Q2_CUDA_PATH_DIR/SHA256SUMS"

echo "[1/6] archive checksums"
sha256sum -c "$SHA256SUMS" >/tmp/llamacpp_32k_kv_sha256_check.out
tail -5 /tmp/llamacpp_32k_kv_sha256_check.out
sha256sum -c "$GRAPH_AB_SHA256SUMS" >/tmp/llamacpp_cuda_graph_ab_sha256_check.out
tail -3 /tmp/llamacpp_cuda_graph_ab_sha256_check.out
sha256sum -c "$GOAL_STATUS_SHA256SUMS" >/tmp/llamacpp_goal_status_sha256_check.out
tail -3 /tmp/llamacpp_goal_status_sha256_check.out
sha256sum -c "$Q2_CUDA_PATH_SHA256SUMS" >/tmp/q2_cuda_path_sha256_check.out
tail -4 /tmp/q2_cuda_path_sha256_check.out

echo "[2/6] matrix conclusions"
python3 scripts/check_llamacpp_kv_matrix.py "$COMBINED_CSV"
python3 scripts/check_cuda_graph_ab.py "$GRAPH_AB_DIR/graph_ab.csv"
python3 scripts/check_q2_cuda_static.py
python3 scripts/check_llamacpp_only.py
python3 scripts/check_kv_cache_types.py
python3 scripts/check_q2_profile_safety.py
python3 scripts/check_build_defaults.py
python3 scripts/check_futuremls_q2_cuda_plan.py
python3 scripts/check_legacy_bench_safety.py
python3 scripts/check_execution_entrypoints.py
python3 scripts/check_32k_matrix_commands.py
python3 scripts/check_32k_ramp_commands.py
python3 scripts/check_q2_ramp_gate.py
python3 scripts/check_q2_ramp_gate_harness_guard.py
python3 scripts/check_q2_ramp_next.py
python3 scripts/check_q2_owht_mean_semantics.py
python3 scripts/check_q2_owht_reader_limits.py
python3 scripts/check_kv_no_hadamard_graph_gate.py
python3 scripts/check_hp_prefill_q2_limits.py
python3 scripts/check_int2_format_boundaries.py
python3 scripts/check_oscar_turbo3_case.py
python3 scripts/check_oscar_turbo3_smoke.py
python3 scripts/check_mixed_k_quality_variant.py
python3 scripts/check_mixed_vec_smoke.py
python3 scripts/check_mixed_vec_int2_ramp.py
python3 scripts/check_mixed_vec_quality_expansion.py
python3 scripts/check_mixed_vec_tiled_decision.py
python3 scripts/check_mixed_tile_boundary.py
python3 scripts/check_q2_fa_onoff_boundary.py
python3 scripts/check_int2_quantizer_comparison.py
python3 scripts/check_int2_attention_error.py
python3 scripts/check_q2_kq_softmax_dump.py
python3 scripts/check_q2_nohad_reader_compat.py
python3 scripts/check_q2_cache_kq_error.py
python3 scripts/check_q2_runtime_dump_hooks.py
python3 scripts/check_oscar_int2_owht_harness.py
python3 scripts/check_llama_completion_direct_smoke.py
python3 scripts/check_q2_logits_path_dump.py
python3 scripts/check_int4_accuracy_boundary.py
python3 scripts/check_no_hadamard_graph_gate_direct_smoke.py
python3 scripts/check_q2_writer_ab.py
python3 scripts/check_q2_top_token_drift.py
python3 scripts/check_layer_drift.py
python3 scripts/check_q2_cache_reconstruction_error.py
python3 scripts/check_q2_quantizer_reconstruction_sweep.py
python3 scripts/check_watchdog_summary_fields.py
python3 scripts/check_post_case_cooldown.py
python3 scripts/check_recovery_readiness.py
python3 scripts/check_turbo2_not_int2.py
.venv-oscar-kv/bin/python scripts/check_granite_rotation_alignment.py
if [[ "${SKIP_NO_GPU_VERIFIER_SELFTEST:-0}" != "1" ]]; then
  python3 scripts/check_no_gpu_verifier_no_nvidia_smi.py
fi
python3 scripts/audit_goal_status.py >/tmp/llamacpp_goal_status.out
python3 scripts/audit_goal_status.py --out-dir /tmp/llamacpp_goal_status_verify >/tmp/llamacpp_goal_status_report.out
if ! rg -q "^overall_status=complete$" /tmp/llamacpp_goal_status.out; then
  cat /tmp/llamacpp_goal_status.out >&2
  echo "expected current deliverable goal audit to be complete" >&2
  exit 1
fi
if ! rg -q "^exact_int2_research_status=incomplete$" /tmp/llamacpp_goal_status.out; then
  cat /tmp/llamacpp_goal_status.out >&2
  echo "expected exact INT2 research status to remain incomplete until 32k q2_0/q2_0 is valid" >&2
  exit 1
fi
if ! rg -q "^32k_turbo2_reference=complete" /tmp/llamacpp_goal_status.out; then
  cat /tmp/llamacpp_goal_status.out >&2
  echo "expected 32k Turbo2 reference to be complete" >&2
  exit 1
fi
if ! rg -q "^32k_int2_speed_target=incomplete" /tmp/llamacpp_goal_status.out; then
  cat /tmp/llamacpp_goal_status.out >&2
  echo "expected exact 32k INT2 speed target to remain incomplete" >&2
  exit 1
fi
if ! cmp -s "$GOAL_STATUS_DIR/goal_status.csv" /tmp/llamacpp_goal_status_verify/goal_status.csv; then
  echo "goal status archive is stale: regenerate with scripts/audit_goal_status.py --out-dir $GOAL_STATUS_DIR" >&2
  exit 1
fi

echo "[3/6] script syntax"
python3 -m py_compile \
  scripts/audit_goal_status.py \
  scripts/check_llamacpp_kv_matrix.py \
  scripts/check_cuda_graph_ab.py \
  scripts/check_q2_cuda_static.py \
  scripts/check_llamacpp_only.py \
  scripts/check_kv_cache_types.py \
  scripts/check_q2_profile_safety.py \
  scripts/check_build_defaults.py \
  scripts/check_futuremls_q2_cuda_plan.py \
  scripts/check_legacy_bench_safety.py \
  scripts/check_32k_matrix_commands.py \
  scripts/check_32k_ramp_commands.py \
  scripts/check_q2_ramp_gate.py \
  scripts/check_q2_ramp_gate_harness_guard.py \
  scripts/check_q2_ramp_next.py \
  scripts/check_q2_owht_mean_semantics.py \
  scripts/check_q2_owht_reader_limits.py \
  scripts/check_kv_no_hadamard_graph_gate.py \
  scripts/check_hp_prefill_q2_limits.py \
  scripts/check_int2_format_boundaries.py \
  scripts/check_oscar_turbo3_case.py \
  scripts/check_oscar_turbo3_smoke.py \
  scripts/check_mixed_k_quality_variant.py \
  scripts/check_mixed_vec_smoke.py \
  scripts/check_mixed_vec_int2_ramp.py \
  scripts/check_mixed_vec_quality_expansion.py \
  scripts/check_mixed_vec_tiled_decision.py \
  scripts/check_mixed_tile_boundary.py \
  scripts/check_q2_fa_onoff_boundary.py \
  scripts/check_int2_quantizer_comparison.py \
  scripts/check_int2_attention_error.py \
  scripts/check_q2_kq_softmax_dump.py \
  scripts/check_q2_nohad_reader_compat.py \
  scripts/check_q2_cache_kq_error.py \
  scripts/check_q2_runtime_dump_hooks.py \
  scripts/check_oscar_int2_owht_harness.py \
  scripts/check_llama_completion_direct_smoke.py \
  scripts/check_q2_logits_path_dump.py \
  scripts/check_int4_accuracy_boundary.py \
  scripts/check_no_hadamard_graph_gate_direct_smoke.py \
  scripts/check_q2_writer_ab.py \
  scripts/check_q2_top_token_drift.py \
  scripts/check_layer_drift.py \
  scripts/check_q2_cache_reconstruction_error.py \
  scripts/check_q2_quantizer_reconstruction_sweep.py \
  scripts/check_execution_entrypoints.py \
  scripts/check_watchdog_summary_fields.py \
  scripts/check_no_gpu_verifier_no_nvidia_smi.py \
  scripts/check_post_case_cooldown.py \
  scripts/check_recovery_readiness.py \
  scripts/check_turbo2_not_int2.py \
  scripts/check_granite_rotation_alignment.py \
  scripts/print_32k_q2_ramp_commands.py \
  scripts/print_32k_matrix_commands.py \
  scripts/refresh_run_archive_checksums.py \
  scripts/report_recovery_readiness.py \
  scripts/report_q2_cuda_path.py \
  scripts/report_q2_ramp_gate.py \
  scripts/run_q2_ramp_next.py \
  scripts/combine_llamacpp_kv_runs.py \
  scripts/analyze_logits_top_tokens.py \
  scripts/analyze_layer_drift.py \
  scripts/analyze_q2_cache_reconstruction_error.py \
  scripts/analyze_q2_cache_kq_error.py \
  scripts/sweep_q2_quantizer_reconstruction.py \
  scripts/summarize_q2_logits_path_dump.py \
  scripts/summarize_cuda_graph_ab.py \
  scripts/summarize_32k_llamacpp_kv.py
bash -n scripts/measure_vram.sh scripts/bench_32k_llamacpp_kv.sh scripts/run_gpqa_gsm8k_kv_eval.sh
if ! rg -q "MAX_PEAK_MIB exceeded" scripts/measure_vram.sh; then
  echo "expected measure_vram to support MAX_PEAK_MIB watchdog" >&2
  exit 1
fi
if ! rg -q "limit_triggered=" scripts/measure_vram.sh; then
  echo "expected measure_vram summaries to record watchdog status" >&2
  exit 1
fi
if ! rg -q "POST_CASE_COOLDOWN_SEC" scripts/bench_32k_llamacpp_kv.sh; then
  echo "expected 32k harness to expose post-case cooldown guard" >&2
  exit 1
fi
if ! rg -q "GPU did not cool down" scripts/bench_32k_llamacpp_kv.sh; then
  echo "expected 32k harness to report cooldown timeout" >&2
  exit 1
fi
bash -n scripts/bench_kv_cache.sh scripts/bench_kv_cache_matrix.sh scripts/run_kv_ppl_matrix.sh
bash -n scripts/cuda_graph_ab.sh
bash -n scripts/q2_profile.sh scripts/q2_segment_bench.sh
rm -rf /tmp/q2_cuda_path_report
python3 scripts/report_q2_cuda_path.py --out-dir /tmp/q2_cuda_path_report >/tmp/q2_cuda_path_report.out
if ! rg -q "KQ q2_0: .*dp4a=3" /tmp/q2_cuda_path_report.out; then
  cat /tmp/q2_cuda_path_report.out >&2
  echo "expected q2 CUDA path report to show q2 KQ 3-dp4a path" >&2
  exit 1
fi
if ! rg -q "KQ q4_0: .*dp4a=1" /tmp/q2_cuda_path_report.out; then
  cat /tmp/q2_cuda_path_report.out >&2
  echo "expected q2 CUDA path report to show q4 KQ 1-dp4a path" >&2
  exit 1
fi
if ! rg -q "V q2_0: .*scalar_decode=yes" /tmp/q2_cuda_path_report.out; then
  cat /tmp/q2_cuda_path_report.out >&2
  echo "expected q2 CUDA path report to show q2 V scalar decode" >&2
  exit 1
fi
if [[ ! -s /tmp/q2_cuda_path_report/q2_cuda_path.csv || ! -s /tmp/q2_cuda_path_report/q2_cuda_path.md ]]; then
  find /tmp/q2_cuda_path_report -maxdepth 1 -type f -printf '%f\n' >&2
  echo "expected q2 CUDA path report files in /tmp" >&2
  exit 1
fi
if ! head -1 /tmp/q2_cuda_path_report/q2_cuda_path.csv | rg -q "fingerprint"; then
  cat /tmp/q2_cuda_path_report/q2_cuda_path.csv >&2
  echo "expected q2 CUDA path report to include function fingerprints" >&2
  exit 1
fi
if ! rg -q "dispatch,KQ,get_vec_dot_KQ" /tmp/q2_cuda_path_report/q2_cuda_path.csv; then
  cat /tmp/q2_cuda_path_report/q2_cuda_path.csv >&2
  echo "expected q2 CUDA path report to include KQ dispatch fingerprint" >&2
  exit 1
fi
if ! cmp -s "$Q2_CUDA_PATH_DIR/q2_cuda_path.csv" /tmp/q2_cuda_path_report/q2_cuda_path.csv; then
  echo "q2 CUDA path archive is stale: regenerate with scripts/report_q2_cuda_path.py --out-dir $Q2_CUDA_PATH_DIR" >&2
  exit 1
fi
if ! rg -q "GGML_CUDA_GRAPHS:BOOL=ON" third_party/OSCAR/build-cuda/CMakeCache.txt; then
  echo "expected build-cuda to have GGML_CUDA_GRAPHS=ON" >&2
  exit 1
fi

echo "[4/6] dry-run safety guards"
DRY_RUN=1 scripts/run_llamacpp.sh >/tmp/run_llamacpp_dry_run.out
if ! rg -q "DRY_RUN command:" /tmp/run_llamacpp_dry_run.out; then
  cat /tmp/run_llamacpp_dry_run.out >&2
  echo "expected run_llamacpp dry-run to render command" >&2
  exit 1
fi
if ! rg -q "Dry run complete; no inference launched." /tmp/run_llamacpp_dry_run.out; then
  cat /tmp/run_llamacpp_dry_run.out >&2
  echo "expected run_llamacpp dry-run to avoid launching inference" >&2
  exit 1
fi
if DRY_RUN=0 scripts/run_llamacpp.sh >/tmp/run_llamacpp_no_ack.out 2>&1; then
  echo "expected run_llamacpp to reject without ACK_RUN_LLAMA=1" >&2
  exit 1
fi
if ! rg -q "ACK_RUN_LLAMA=1" /tmp/run_llamacpp_no_ack.out; then
  cat /tmp/run_llamacpp_no_ack.out >&2
  echo "missing run_llamacpp ACK guard message" >&2
  exit 1
fi
rm -rf /tmp/legacy_bench_guard_runs
DRY_RUN=1 RUNS_DIR=/tmp/legacy_bench_guard_runs scripts/bench_kv_cache.sh >/tmp/legacy_bench_dry_run.out
if ! rg -q "DRY_RUN command:" /tmp/legacy_bench_dry_run.out; then
  cat /tmp/legacy_bench_dry_run.out >&2
  echo "expected legacy bench dry-run to render commands" >&2
  exit 1
fi
if ! rg -q "Dry run complete; no results written." /tmp/legacy_bench_dry_run.out; then
  cat /tmp/legacy_bench_dry_run.out >&2
  echo "expected legacy bench dry-run to avoid writing results" >&2
  exit 1
fi
if [[ -d /tmp/legacy_bench_guard_runs ]]; then
  find /tmp/legacy_bench_guard_runs -maxdepth 2 -type f -printf '%p\n' >&2
  echo "legacy bench dry-run unexpectedly wrote files" >&2
  exit 1
fi
if DRY_RUN=0 RUNS_DIR=/tmp/legacy_bench_guard_runs scripts/bench_kv_cache.sh >/tmp/legacy_bench_no_ack.out 2>&1; then
  echo "expected legacy bench heavy settings to reject without ACK_HEAVY_CONTEXT=1" >&2
  exit 1
fi
if ! rg -q "ACK_HEAVY_CONTEXT=1" /tmp/legacy_bench_no_ack.out; then
  cat /tmp/legacy_bench_no_ack.out >&2
  echo "missing legacy bench heavy ACK guard message" >&2
  exit 1
fi
rm -rf /tmp/legacy_matrix_guard_runs
DRY_RUN=1 RUNS_DIR=/tmp/legacy_matrix_guard_runs scripts/bench_kv_cache_matrix.sh >/tmp/legacy_matrix_dry_run.out
if ! rg -q "DRY_RUN command:" /tmp/legacy_matrix_dry_run.out; then
  cat /tmp/legacy_matrix_dry_run.out >&2
  echo "expected matrix bench dry-run to render commands" >&2
  exit 1
fi
if ! rg -q "no preflight, GPU checks, or results written" /tmp/legacy_matrix_dry_run.out; then
  cat /tmp/legacy_matrix_dry_run.out >&2
  echo "expected matrix bench dry-run to avoid preflight/GPU checks/results" >&2
  exit 1
fi
if [[ -d /tmp/legacy_matrix_guard_runs ]]; then
  find /tmp/legacy_matrix_guard_runs -maxdepth 2 -type f -printf '%p\n' >&2
  echo "matrix bench dry-run unexpectedly wrote files" >&2
  exit 1
fi
if DRY_RUN=0 RUNS_DIR=/tmp/legacy_matrix_guard_runs scripts/bench_kv_cache_matrix.sh >/tmp/legacy_matrix_no_ack.out 2>&1; then
  echo "expected matrix bench to reject without ACK_MATRIX_BENCH=1" >&2
  exit 1
fi
if ! rg -q "ACK_MATRIX_BENCH=1" /tmp/legacy_matrix_no_ack.out; then
  cat /tmp/legacy_matrix_no_ack.out >&2
  echo "missing matrix bench ACK guard message" >&2
  exit 1
fi
rm -rf /tmp/ppl_matrix_guard_runs
DRY_RUN=1 RUNS_DIR=/tmp/ppl_matrix_guard_runs scripts/run_kv_ppl_matrix.sh >/tmp/ppl_matrix_dry_run.out
if ! rg -q "DRY_RUN command:" /tmp/ppl_matrix_dry_run.out; then
  cat /tmp/ppl_matrix_dry_run.out >&2
  echo "expected PPL matrix dry-run to render commands" >&2
  exit 1
fi
if ! rg -q "no executable checks, corpus checks, preflight, GPU checks, or results written" /tmp/ppl_matrix_dry_run.out; then
  cat /tmp/ppl_matrix_dry_run.out >&2
  echo "expected PPL matrix dry-run to avoid executable/corpus/preflight/GPU checks/results" >&2
  exit 1
fi
if [[ -d /tmp/ppl_matrix_guard_runs ]]; then
  find /tmp/ppl_matrix_guard_runs -maxdepth 2 -type f -printf '%p\n' >&2
  echo "PPL matrix dry-run unexpectedly wrote files" >&2
  exit 1
fi
if DRY_RUN=0 RUNS_DIR=/tmp/ppl_matrix_guard_runs scripts/run_kv_ppl_matrix.sh >/tmp/ppl_matrix_no_ack.out 2>&1; then
  echo "expected PPL matrix to reject without ACK_PPL_MATRIX=1" >&2
  exit 1
fi
if ! rg -q "ACK_PPL_MATRIX=1" /tmp/ppl_matrix_no_ack.out; then
  cat /tmp/ppl_matrix_no_ack.out >&2
  echo "missing PPL matrix ACK guard message" >&2
  exit 1
fi
rm -rf /tmp/llamacpp_32k_default_dry_run
RUNS_DIR=/tmp/llamacpp_32k_default_dry_run scripts/bench_32k_llamacpp_kv.sh >/tmp/llamacpp_32k_default_dry_run.out
if ! rg -q "DRY_RUN command:" /tmp/llamacpp_32k_default_dry_run.out; then
  cat /tmp/llamacpp_32k_default_dry_run.out >&2
  echo "expected 32k harness default to dry-run command rendering" >&2
  exit 1
fi
if ! rg -q "Dry run complete; no results written." /tmp/llamacpp_32k_default_dry_run.out; then
  cat /tmp/llamacpp_32k_default_dry_run.out >&2
  echo "expected 32k harness default dry-run to avoid writing results" >&2
  exit 1
fi
if [[ -d /tmp/llamacpp_32k_default_dry_run ]]; then
  find /tmp/llamacpp_32k_default_dry_run -maxdepth 2 -type f -printf '%p\n' >&2
  echo "32k harness default dry-run unexpectedly wrote files" >&2
  exit 1
fi
if PROMPT_TOKENS=32768 CASES=all DRY_RUN=1 ACK_HEAVY_32K=1 ACK_Q2_32K_NOGO=1 \
    scripts/bench_32k_llamacpp_kv.sh >/tmp/llamacpp_32k_guard_all.out 2>&1; then
  echo "expected CASES=all q2 guard to reject" >&2
  exit 1
fi
if ! rg -q "multi-case" /tmp/llamacpp_32k_guard_all.out; then
  cat /tmp/llamacpp_32k_guard_all.out >&2
  echo "missing multi-case q2 guard message" >&2
  exit 1
fi
if PROMPT_TOKENS=32768 CASES=turbo2_streamk GEN_TOKENS=64 DRY_RUN=1 \
    ACK_HEAVY_32K=1 ACK_Q2_32K_NOGO=1 \
    scripts/bench_32k_llamacpp_kv.sh >/tmp/llamacpp_32k_guard_turbo2_decode.out 2>&1; then
  echo "expected GEN_TOKENS turbo2 guard to reject" >&2
  exit 1
fi
if ! rg -q "GEN_TOKENS=1 and REPETITIONS=1" /tmp/llamacpp_32k_guard_turbo2_decode.out; then
  cat /tmp/llamacpp_32k_guard_turbo2_decode.out >&2
  echo "missing decode/repetition turbo2 guard message" >&2
  exit 1
fi
if PROMPT_TOKENS=32768 CASES=oscar_int2 GEN_TOKENS=64 DRY_RUN=1 \
    ACK_HEAVY_32K=1 ACK_Q2_32K_NOGO=1 \
    scripts/bench_32k_llamacpp_kv.sh >/tmp/llamacpp_32k_guard_decode.out 2>&1; then
  echo "expected GEN_TOKENS q2 guard to reject" >&2
  exit 1
fi
if ! rg -q "GEN_TOKENS=1 and REPETITIONS=1" /tmp/llamacpp_32k_guard_decode.out; then
  cat /tmp/llamacpp_32k_guard_decode.out >&2
  echo "missing decode/repetition q2 guard message" >&2
  exit 1
fi
PROMPT_TOKENS=32768 CASES=oscar_int2 GEN_TOKENS=1 REPETITIONS=1 DRY_RUN=1 MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  ACK_HEAVY_32K=1 ACK_Q2_32K_NOGO=1 CUDA_GRAPHS_MODE=on CUDA_GRAPH_OPT=1 \
  scripts/bench_32k_llamacpp_kv.sh >/tmp/llamacpp_32k_guard_allowed.out
if ! rg -q "DRY_RUN command:" /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected single-case q2 dry-run to reach command rendering" >&2
  exit 1
fi
if ! rg -q "Dry run complete; no results written." /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected dry-run to avoid writing results" >&2
  exit 1
fi
if find runs -maxdepth 1 -type d -name 'llamacpp_32k_kv_dry_run' | rg -q .; then
  echo "dry-run unexpectedly wrote under runs/" >&2
  exit 1
fi
if ! rg -q "GGML_CUDA_GRAPH_OPT=1" /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected graph optimization env in dry-run command" >&2
  exit 1
fi
if ! rg -q "GGML_CUDA_DISABLE_GRAPHS=" /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected CUDA graph enable env in dry-run command" >&2
  exit 1
fi
if ! rg -q "MAX_PEAK_MIB=7000" /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected peak-VRAM watchdog env in dry-run command" >&2
  exit 1
fi
if ! rg -q "POST_CASE_COOLDOWN_SEC=30" /tmp/llamacpp_32k_guard_allowed.out; then
  cat /tmp/llamacpp_32k_guard_allowed.out >&2
  echo "expected post-case cooldown in dry-run output" >&2
  exit 1
fi
rm -rf /tmp/q2_profile_dry_run /tmp/q2_segments_dry_run
DRY_RUN=1 OUT_DIR=/tmp/q2_profile_dry_run scripts/q2_profile.sh >/tmp/q2_profile_dry_run.out
if ! rg -q "build-cuda/bin/llama-bench" /tmp/q2_profile_dry_run/dry_run.txt; then
  cat /tmp/q2_profile_dry_run/dry_run.txt >&2
  echo "expected q2_profile dry-run to use build-cuda llama-bench" >&2
  exit 1
fi
if find /tmp/q2_profile_dry_run -maxdepth 1 -type f \
    \( -name 'preflight_summary.txt' -o -name 'preflight.log' -o -name 'ncu.log' -o -name 'nsys.log' -o -name 'fallback.log' \) | rg -q .; then
  find /tmp/q2_profile_dry_run -maxdepth 1 -type f -printf '%f\n' >&2
  echo "q2_profile dry-run unexpectedly produced profiler/fallback logs" >&2
  exit 1
fi
DRY_RUN=1 OUT_DIR=/tmp/q2_segments_dry_run scripts/q2_segment_bench.sh >/tmp/q2_segments_dry_run.out
if ! rg -q "build-cuda/bin/llama-bench" /tmp/q2_segments_dry_run/dry_run.txt; then
  cat /tmp/q2_segments_dry_run/dry_run.txt >&2
  echo "expected q2_segment_bench dry-run to use build-cuda llama-bench" >&2
  exit 1
fi
if find /tmp/q2_segments_dry_run -maxdepth 1 -type f \
    \( -name '*.log' -o -name 'summary.json' \) | rg -q .; then
  find /tmp/q2_segments_dry_run -maxdepth 1 -type f -printf '%f\n' >&2
  echo "q2_segment_bench dry-run unexpectedly produced benchmark logs" >&2
  exit 1
fi
scripts/cuda_graph_ab.sh >/tmp/llamacpp_cuda_graph_ab_dry_run.out
if ! rg -q "CUDA graph mode=off opt=0" /tmp/llamacpp_cuda_graph_ab_dry_run.out; then
  cat /tmp/llamacpp_cuda_graph_ab_dry_run.out >&2
  echo "expected graph A/B dry-run off mode" >&2
  exit 1
fi
if ! rg -q "CUDA graph mode=on opt=1" /tmp/llamacpp_cuda_graph_ab_dry_run.out; then
  cat /tmp/llamacpp_cuda_graph_ab_dry_run.out >&2
  echo "expected graph A/B dry-run on mode" >&2
  exit 1
fi
if ! rg -q "Dry run complete; set RUN_REAL=1" /tmp/llamacpp_cuda_graph_ab_dry_run.out; then
  cat /tmp/llamacpp_cuda_graph_ab_dry_run.out >&2
  echo "expected graph A/B helper to default to dry-run" >&2
  exit 1
fi
if [[ -d runs/cuda_graph_ab_20260612T062854Z ]]; then
  python3 scripts/summarize_cuda_graph_ab.py runs/cuda_graph_ab_20260612T062854Z >/tmp/llamacpp_cuda_graph_ab_summary.out
  if ! rg -q "graph_ab.csv" /tmp/llamacpp_cuda_graph_ab_summary.out; then
    cat /tmp/llamacpp_cuda_graph_ab_summary.out >&2
    echo "expected graph A/B summary output" >&2
    exit 1
  fi
fi

echo "[5/6] forbidden harness keywords"
python3 scripts/check_llamacpp_only.py

echo "[6/6] GPU idle snapshot"
if [[ "$CHECK_GPU_SNAPSHOT" != "1" ]]; then
  echo "skipped; set CHECK_GPU_SNAPSHOT=1 for a read-only nvidia-smi snapshot"
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits
else
  echo "nvidia-smi not found; skipping GPU snapshot"
fi

echo "no-GPU verification passed"
