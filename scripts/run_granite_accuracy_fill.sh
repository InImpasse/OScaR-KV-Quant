#!/usr/bin/env bash
# Fill the remaining Granite accuracy baseline:
#   1) HumanEval OSCAR INT2, n=164
#   2) MATH-500 BF16 / plain INT2 / OSCAR INT2, n=500
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${ROOT}"
# shellcheck source=lib/repo_paths.sh
source "${ROOT}/scripts/lib/repo_paths.sh"

MODEL="checkpoints/granite-4.0-1b-base"
ROT_DIR="rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
TAG="$(date +%Y%m%dT%H%M%S)"
GPU="0"
HUMANEVAL_N="164"
MATH_N="500"
PY=".venv-oscar-kv/bin/python"
MAX_NEW_TOKENS="2048"
NUM_THREADS="1"
REPEAT="1"
POST_READY_SLEEP="3"
SUMMARY_ROOT="rotation/granite-4.0-1b/accuracy_fill"
ONLY="all"

usage() {
  cat <<'EOF'
Usage: scripts/run_granite_accuracy_fill.sh [options]

Runs only the missing Granite accuracy jobs:
  - HumanEval oscar-int2, 164 examples
  - MATH-500 bf16 / int2 / oscar-int2, 500 examples

Options:
  --model PATH             Model checkpoint directory
  --rot-dir PATH           OSCAR rotation directory
  --tag TAG                Output tag (default: timestamp)
  --gpu N                  CUDA device index (default: 0)
  --humaneval-n N          HumanEval examples (default: 164)
  --math-n N               MATH examples (default: 500)
  --summary-root PATH      Parent summary directory
  --max-new-tokens N       Generation length (default: 2048)
  --num-threads N          Eval parallelism (default: 1)
  --repeat N               Repeat count (default: 1)
  --post-ready-sleep N     Delay after server readiness (default: 3)
  --only TARGET            all | math | humaneval (default: all)
  -h, --help               Show this help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --rot-dir) ROT_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --humaneval-n) HUMANEVAL_N="$2"; shift 2 ;;
    --math-n) MATH_N="$2"; shift 2 ;;
    --summary-root) SUMMARY_ROOT="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --post-ready-sleep) POST_READY_SLEEP="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
done

case "${ONLY}" in
  all|math|humaneval) ;;
  *) echo "invalid --only ${ONLY}; use all, math, or humaneval" >&2; exit 2 ;;
esac

MODEL="$(resolve_repo_path "${MODEL}")"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
PY="$(resolve_repo_path "${PY}")"
SUMMARY_DIR="$(resolve_repo_path "${SUMMARY_ROOT}")/${TAG}"
SCRIPT_DIR="${ROOT}/rotation/granite-4.0-1b"
mkdir -p "${SUMMARY_DIR}"

MAIN_LOG="${SUMMARY_DIR}/fill.log"
MANIFEST="${SUMMARY_DIR}/manifest.tsv"
: > "${MAIN_LOG}"
printf 'task\tmode\tn\trun_dir\tlog\tmetrics\n' > "${MANIFEST}"

log() {
  echo "[accuracy_fill ${TAG}] $*" | tee -a "${MAIN_LOG}"
}

record_manifest() {
  local task="$1"
  local mode="$2"
  local n="$3"
  local run_dir="$4"
  local step_log="$5"
  local metrics="${run_dir}/metrics.json"
  if [[ ! -f "${metrics}" ]]; then
    metrics=""
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task}" "${mode}" "${n}" \
    "$(repo_relative_path "${run_dir}")" \
    "$(repo_relative_path "${step_log}")" \
    "${metrics:+$(repo_relative_path "${metrics}")}" >> "${MANIFEST}"
}

run_simple() {
  local task="$1"
  local mode="$2"
  local n="$3"
  local port="$4"
  local task_upper
  task_upper="$(echo "${task}" | tr '[:lower:]' '[:upper:]')"
  local run_dir="${SCRIPT_DIR}/${task_upper}/eval_${mode}_n${n}_${TAG}"
  local step_log="${SUMMARY_DIR}/${task}_${mode}.log"

  log "start task=${task} mode=${mode} n=${n} run_dir=$(repo_relative_path "${run_dir}")"
  bash "${SCRIPT_DIR}/eval_simple_suite_granite.sh" \
    --task "${task}" \
    --mode "${mode}" \
    --num-examples "${n}" \
    --model "${MODEL}" \
    --rot-dir "${ROT_DIR}" \
    --run-dir "${run_dir}" \
    --port "${port}" \
    --gpu "${GPU}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-threads "${NUM_THREADS}" \
    --repeat "${REPEAT}" \
    --post-ready-sleep "${POST_READY_SLEEP}" \
    2>&1 | tee "${step_log}"
  record_manifest "${task}" "${mode}" "${n}" "${run_dir}" "${step_log}"
  log "done task=${task} mode=${mode}"
}

compare_three_modes() {
  local task="$1"
  local n="$2"
  local task_upper
  task_upper="$(echo "${task}" | tr '[:lower:]' '[:upper:]')"
  local out_json="${SUMMARY_DIR}/${task}_compare.json"
  local bf16_dir="${SCRIPT_DIR}/${task_upper}/eval_bf16_n${n}_${TAG}"
  local int2_dir="${SCRIPT_DIR}/${task_upper}/eval_int2_n${n}_${TAG}"
  local oscar_dir="${SCRIPT_DIR}/${task_upper}/eval_oscar-int2_n${n}_${TAG}"

  if [[ -f "${bf16_dir}/metrics.json" && -f "${int2_dir}/metrics.json" && -f "${oscar_dir}/metrics.json" ]]; then
    log "compare task=${task} -> $(repo_relative_path "${out_json}")"
    "${PY}" -m oscar_kv_quant.accuracy_compare \
      --bf16 "${bf16_dir}" \
      --int2 "${int2_dir}" \
      --oscar-int2 "${oscar_dir}" \
      --output "${out_json}" | tee -a "${MAIN_LOG}"
  else
    log "skip compare task=${task}: one or more metrics.json files are missing"
  fi
}

log "model=$(repo_relative_path "${MODEL}")"
log "rot_dir=$(repo_relative_path "${ROT_DIR}")"
log "summary_dir=$(repo_relative_path "${SUMMARY_DIR}")"
log "settings gpu=${GPU} only=${ONLY} humaneval_n=${HUMANEVAL_N} math_n=${MATH_N} max_new_tokens=${MAX_NEW_TOKENS}"

if [[ "${ONLY}" == "all" || "${ONLY}" == "humaneval" ]]; then
  run_simple "humaneval" "oscar-int2" "${HUMANEVAL_N}" 33320
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "math" ]]; then
  run_simple "math" "bf16" "${MATH_N}" 33330
  run_simple "math" "int2" "${MATH_N}" 33331
  run_simple "math" "oscar-int2" "${MATH_N}" 33332
  compare_three_modes "math" "${MATH_N}"
fi

log "done"
log "manifest=$(repo_relative_path "${MANIFEST}")"
