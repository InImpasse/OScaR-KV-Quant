#!/usr/bin/env bash
# Run llama.cpp preset matrix with CUDA graph off vs on.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=llamacpp_matrix_common.sh
source "$ROOT_DIR/scripts/llamacpp_matrix_common.sh"

RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$RUNS_DIR/llamacpp_cuda_graph_compare_matrix_$STAMP}"
PRESETS="${PRESETS:-short,medium,long,16k,32k}"
MODES="${MODES:-bf16,oscar-int2,int2}"
GEN_TOKENS="${GEN_TOKENS:-1}"
REPETITIONS="${REPETITIONS:-1}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-180}"
RUN_REAL="${RUN_REAL:-0}"
ACK_HEAVY_32K="${ACK_HEAVY_32K:-0}"
ACK_Q2_32K_NOGO="${ACK_Q2_32K_NOGO:-0}"
ACK_Q2_RAMP_GATE_HOLD="${ACK_Q2_RAMP_GATE_HOLD:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/cuda_graph_compare_llamacpp_matrix.sh [options]

Runs the llama.cpp preset matrix for CUDA graph off and on. Defaults to dry-run.

Options:
  --out-root DIR       Output parent directory
  --presets LIST      Comma list: short,medium,long,16k,32k
  --modes LIST        Comma list: bf16,int2,oscar-int2,int4,oscar-int4
  --gen-tokens N      llama-bench -n value
  --case-timeout SEC  Per-case timeout
  --real              Execute benchmarks (same as RUN_REAL=1)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --presets) PRESETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --case-timeout) CASE_TIMEOUT_SEC="$2"; shift 2 ;;
    --real) RUN_REAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

dry_run=1
if [[ "$RUN_REAL" == "1" ]]; then
  dry_run=0
fi

for graph_state in off on; do
  opt=0
  [[ "$graph_state" == "on" ]] && opt=1
  echo "[cuda_graph_compare_llamacpp_matrix] graph=$graph_state opt=$opt"
  OUT_ROOT="$OUT_ROOT/$graph_state" \
  PRESETS="$PRESETS" \
  MODES="$MODES" \
  GEN_TOKENS="$GEN_TOKENS" \
  REPETITIONS="$REPETITIONS" \
  CASE_TIMEOUT_SEC="$CASE_TIMEOUT_SEC" \
  DRY_RUN="$dry_run" \
  CUDA_GRAPHS_MODE="$graph_state" \
  CUDA_GRAPH_OPT="$opt" \
  ACK_HEAVY_32K="$ACK_HEAVY_32K" \
  ACK_Q2_32K_NOGO="$ACK_Q2_32K_NOGO" \
  ACK_Q2_RAMP_GATE_HOLD="$ACK_Q2_RAMP_GATE_HOLD" \
    "$ROOT_DIR/scripts/bench_llamacpp_matrix.sh"
done

if [[ "$RUN_REAL" == "1" ]]; then
  python3 "$ROOT_DIR/scripts/summarize_llamacpp_matrix.py" "$OUT_ROOT" --graph-compare
  echo "[cuda_graph_compare_llamacpp_matrix] results -> $OUT_ROOT"
else
  echo "[cuda_graph_compare_llamacpp_matrix] dry run complete; pass --real or RUN_REAL=1 to execute."
fi
