#!/usr/bin/env bash
# Run BF16 / plain INT2 / OSCAR INT2 accuracy evals on Granite benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
# shellcheck source=../../scripts/lib/eval_cli.sh
source "${REPO_ROOT}/scripts/lib/eval_cli.sh"
PY="${PY:-.venv-oscar-kv/bin/python}"

MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/GPQA/seq30000_prompt118_group128/rotations}"
SUITE_TAG="${SUITE_TAG:-$(date +%Y%m%dT%H%M%S)}"
QUICK_SUITE=0
GPQA_N="${GPQA_NUM_EXAMPLES:-198}"
GSM8K_N="${GSM8K_NUM_QUESTIONS:-200}"
HUMANEVAL_N="${HUMANEVAL_NUM_EXAMPLES:-164}"
MATH_N="${MATH_NUM_EXAMPLES:-500}"

EVAL_CLI_HELP_FN=eval_cli_usage_suite
parse_eval_cli_args "$@"
if [[ "${QUICK_SUITE:-0}" == "1" ]]; then
  GPQA_N=0
  GSM8K_N=0
  HUMANEVAL_N=20
  MATH_N=20
fi
MODEL="$(resolve_repo_path "${MODEL}")"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
PY="$(resolve_repo_path "${PY}")"

SUMMARY_DIR="${SCRIPT_DIR}/accuracy_suite_${SUITE_TAG}"
mkdir -p "${SUMMARY_DIR}"
MAIN_LOG="${SUMMARY_DIR}/suite.log"

log() {
  echo "[suite ${SUITE_TAG}] $*" | tee -a "${MAIN_LOG}"
}

run_gpqa_mode() {
  local mode="$1"
  local out="${SCRIPT_DIR}/GPQA/eval_${mode}_suite_${SUITE_TAG}"
  local port
  case "${mode}" in
    bf16) port=31140 ;;
    int2) port=31150 ;;
    oscar-int2) port=31120 ;;
    *) echo "unknown gpqa mode: ${mode}" >&2; return 1 ;;
  esac
  local -a args=(--num-examples "${GPQA_N}" --model "${MODEL}" --port "${port}" --run-dir "${out}")
  case "${mode}" in
    bf16) bash "${SCRIPT_DIR}/eval_gpqa_granite_bf16.sh" "${args[@]}" ;;
    int2) bash "${SCRIPT_DIR}/eval_gpqa_granite_int2.sh" "${args[@]}" ;;
    oscar-int2)
      args+=(--rot-dir "${ROT_DIR}")
      bash "${SCRIPT_DIR}/eval_gpqa_granite.sh" "${args[@]}"
      ;;
  esac
  echo "${out}" > "${SUMMARY_DIR}/gpqa_${mode}.dir"
}

run_gsm8k_mode() {
  local mode="$1"
  local out="${SCRIPT_DIR}/GSM8K/eval_${mode}_q${GSM8K_N}_${SUITE_TAG}"
  local port=$((32200 + $(echo "${mode}" | cksum | awk '{print $1 % 10}')))
  bash "${SCRIPT_DIR}/eval_gsm8k_granite.sh" \
    --mode "${mode}" \
    --num-questions "${GSM8K_N}" \
    --model "${MODEL}" \
    --rot-dir "${ROT_DIR}" \
    --port "${port}" \
    --run-dir "${out}"
  echo "${out}" > "${SUMMARY_DIR}/gsm8k_${mode}.dir"
}

run_simple_task() {
  local task="$1"
  local n="$2"
  local mode="$3"
  local task_upper
  task_upper="$(echo "${task}" | tr '[:lower:]' '[:upper:]')"
  local out="${SCRIPT_DIR}/${task_upper}/eval_${mode}_n${n}_${SUITE_TAG}"
  local port=$((33000 + $(echo "${task}_${mode}" | cksum | awk '{print $1 % 100}')))
  bash "${SCRIPT_DIR}/eval_simple_suite_granite.sh" \
    --task "${task}" \
    --mode "${mode}" \
    --num-examples "${n}" \
    --model "${MODEL}" \
    --rot-dir "${ROT_DIR}" \
    --port "${port}" \
    --run-dir "${out}" \
    --num-threads 1 \
    --repeat 1
  echo "${out}" > "${SUMMARY_DIR}/${task}_${mode}.dir"
}

compare_task() {
  local label="$1"
  local bf16_dir int2_dir oscar_dir out_json
  bf16_dir="$(cat "${SUMMARY_DIR}/${label}_bf16.dir")"
  int2_dir="$(cat "${SUMMARY_DIR}/${label}_int2.dir")"
  oscar_dir="$(cat "${SUMMARY_DIR}/${label}_oscar-int2.dir")"
  out_json="${SUMMARY_DIR}/${label}_compare.json"
  "${PY}" -m oscar_kv_quant.accuracy_compare \
    --bf16 "${bf16_dir}" \
    --int2 "${int2_dir}" \
    --oscar-int2 "${oscar_dir}" \
    --output "${out_json}" | tee -a "${MAIN_LOG}"
}

log "model=${MODEL} rot=${ROT_DIR}"
log "sizes: gpqa=${GPQA_N} gsm8k=${GSM8K_N} humaneval=${HUMANEVAL_N} math=${MATH_N}"

if [[ "${GPQA_N}" != "0" ]]; then
  for mode in bf16 int2 oscar-int2; do
    log "GPQA ${mode} ..."
    run_gpqa_mode "${mode}"
  done
  compare_task "gpqa"
fi

if [[ "${GSM8K_N}" != "0" ]]; then
  for mode in bf16 int2 oscar-int2; do
    log "GSM8K ${mode} ..."
    run_gsm8k_mode "${mode}"
  done
  compare_task "gsm8k"
fi

if [[ "${HUMANEVAL_N}" != "0" ]]; then
  for mode in bf16 int2 oscar-int2; do
    log "HumanEval ${mode} ..."
    run_simple_task "humaneval" "${HUMANEVAL_N}" "${mode}"
  done
  compare_task "humaneval"
fi

if [[ "${MATH_N}" != "0" ]]; then
  for mode in bf16 int2 oscar-int2; do
    log "MATH-500 ${mode} ..."
    run_simple_task "math" "${MATH_N}" "${mode}"
  done
  compare_task "math"
fi

log "done -> ${SUMMARY_DIR}"
