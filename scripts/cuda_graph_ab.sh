#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
CASES="${CASES:-plain_int2}"
GEN_TOKENS="${GEN_TOKENS:-1}"
REPETITIONS="${REPETITIONS:-1}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-60}"
RUN_REAL="${RUN_REAL:-0}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if (( PROMPT_TOKENS > 8192 )) && [[ "$RUN_REAL" != "1" ]]; then
  echo "Dry-run only for PROMPT_TOKENS=$PROMPT_TOKENS. Set RUN_REAL=1 only after a smaller run is healthy." >&2
fi

if (( PROMPT_TOKENS >= 32768 )); then
  echo "Refusing 32k CUDA graph A/B in this low-risk helper. Use bench_32k_llamacpp_kv.sh manually with the q2 guards." >&2
  exit 2
fi

if [[ "$CASES" == "all" || "$CASES" == *,* ]]; then
  echo "Refusing multi-case CUDA graph A/B. Use one CASES value at a time." >&2
  exit 2
fi

if (( GEN_TOKENS != 1 || REPETITIONS != 1 )); then
  echo "Refusing CUDA graph A/B unless GEN_TOKENS=1 and REPETITIONS=1." >&2
  exit 2
fi

dry_run=1
if [[ "$RUN_REAL" == "1" ]]; then
  dry_run=0
  OUT_ROOT="${OUT_ROOT:-$RUNS_DIR/cuda_graph_ab_$STAMP}"
  mkdir -p "$OUT_ROOT"
else
  OUT_ROOT="${OUT_ROOT:-/tmp/cuda_graph_ab_dry_run}"
fi

run_mode() {
  local mode="$1"
  local opt="$2"
  local out_dir="$OUT_ROOT/${mode}_opt${opt}"

  echo
  echo "=== CUDA graph mode=$mode opt=$opt prompt=$PROMPT_TOKENS cases=$CASES ==="
  OUT_DIR="$out_dir" \
  PROMPT_TOKENS="$PROMPT_TOKENS" \
  CASES="$CASES" \
  GEN_TOKENS="$GEN_TOKENS" \
  REPETITIONS="$REPETITIONS" \
  CASE_TIMEOUT_SEC="$CASE_TIMEOUT_SEC" \
  DRY_RUN="$dry_run" \
  CUDA_GRAPHS_MODE="$mode" \
  CUDA_GRAPH_OPT="$opt" \
    "$ROOT_DIR/scripts/bench_32k_llamacpp_kv.sh"
}

run_mode off 0
run_mode on 1

if [[ "$RUN_REAL" == "1" ]]; then
  python3 "$ROOT_DIR/scripts/summarize_cuda_graph_ab.py" "$OUT_ROOT"
  echo
  echo "Results written under: $OUT_ROOT"
else
  echo
  echo "Dry run complete; set RUN_REAL=1 for the low-risk real A/B."
fi
