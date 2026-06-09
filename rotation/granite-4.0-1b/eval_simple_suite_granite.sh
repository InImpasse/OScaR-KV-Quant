#!/usr/bin/env bash
# Run SGLang simple-evals tasks for Granite 4.0 1B across BF16, plain INT2,
# and OSCAR INT2 KV modes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
# shellcheck source=../../scripts/lib/eval_cli.sh
source "${REPO_ROOT}/scripts/lib/eval_cli.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"

MODE="bf16"  # bf16 | int2 | oscar-int2
TASK="humaneval"  # humaneval | aime25 | math | gpqa | gsm8k
MODEL="checkpoints/granite-4.0-1b-base"
ROT_DIR="${SCRIPT_DIR}/GPQA/seq30000_prompt118_group128/rotations"
RUN_DIR=""
PY=".venv-oscar-kv/bin/python"
GPU="0"
PORT="32300"
DIST_PORT="42300"
MEM_FRACTION_STATIC="0.78"
# Honor env (e.g. MAX_RUNNING_REQUESTS=8 bash …) so client-side --repeat
# parallelism does not overwhelm the server's waiting queue (503 queue full).
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-4}"
NUM_EXAMPLES=""
NUM_THREADS="1"
REPEAT="1"
MAX_NEW_TOKENS="2048"
POST_READY_SLEEP="3"
API="completion"
TEMPERATURE="0.0"
TOP_P="1.0"
K_ROT_FILENAME="k_rotation_qqt_r_h_pbr.pt"
V_ROT_FILENAME="v_rotation_sst_r_h_pbr.pt"
MIXED_KV_HP_MAX_SPLITS="2"
MIXED_KV_PREFIX_TOKENS="64"
MIXED_KV_RECENT_TOKENS="256"
MIXED_KV_MAX_QUANT_TOKENS="32768"
MIXED_KV_HP_PREFIX_POOL_TOKENS="1024"
MIXED_KV_SCALE_DTYPE="bfloat16"
OSCAR_K_CLIP_RATIO="0.96"
OSCAR_V_CLIP_RATIO="0.92"
LLOYD_MAX="0"
OSCAR_FUSED_ROTATE_CLIP_QUANT="1"

EVAL_CLI_HELP_FN=eval_cli_usage_simple
parse_eval_cli_args "$@"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/${TASK^^}/eval_${MODE}_$(date +%Y%m%dT%H%M%S)}"
MODEL="$(resolve_repo_path "${MODEL}")"
MODEL_REL="$(repo_relative_path "${MODEL}")"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
PY="$(resolve_repo_path "${PY}")"

if [[ ! -x "${PY}" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[eval simple granite] missing model config: ${MODEL}/config.json" >&2
  exit 1
fi

case "${MODE}" in
  bf16)
    KV_DTYPE="bf16"
    PAGE_SIZE=128
    MODE_ENV=()
    ;;
  int2)
    KV_DTYPE="int2"
    PAGE_SIZE=128
    MODE_ENV=()
    ;;
  oscar-int2)
    KV_DTYPE="int2"
    PAGE_SIZE=8
    K_ROT="${ROT_DIR}/${K_ROT_FILENAME}"
    V_ROT="${ROT_DIR}/${V_ROT_FILENAME}"
    if [[ ! -f "${K_ROT}" || ! -f "${V_ROT}" ]]; then
      echo "[eval simple granite] missing rotation files under ROT_DIR=${ROT_DIR}" >&2
      exit 1
    fi
    MODE_ENV=(
      SGLANG_ENABLE_MIXED_KV_WINDOWS=1
      SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
      SGLANG_OSCAR_ABSORB_V_ROTATION=1
      SGLANG_MIXED_KV_HP_MAX_SPLITS="${MIXED_KV_HP_MAX_SPLITS}"
      SGLANG_MIXED_KV_PREFIX_TOKENS="${MIXED_KV_PREFIX_TOKENS}"
      SGLANG_MIXED_KV_RECENT_TOKENS="${MIXED_KV_RECENT_TOKENS}"
      SGLANG_MIXED_KV_MAX_QUANT_TOKENS="${MIXED_KV_MAX_QUANT_TOKENS}"
      SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${MIXED_KV_HP_PREFIX_POOL_TOKENS}"
      SGLANG_MIXED_KV_HP_DTYPE=bfloat16
      SGLANG_MIXED_KV_SCALE_DTYPE="${MIXED_KV_SCALE_DTYPE}"
      SGLANG_OSCAR_K_ROTATION_PATH="${K_ROT}"
      SGLANG_OSCAR_V_ROTATION_PATH="${V_ROT}"
      SGLANG_OSCAR_K_CLIP_RATIO="${OSCAR_K_CLIP_RATIO}"
      SGLANG_OSCAR_V_CLIP_RATIO="${OSCAR_V_CLIP_RATIO}"
      SGLANG_LLOYD_MAX="${LLOYD_MAX}"
      SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT="${OSCAR_FUSED_ROTATE_CLIP_QUANT}"
    )
    ;;
  *)
    echo "[eval simple granite] unknown MODE=${MODE}; use bf16, int2, or oscar-int2" >&2
    exit 1
    ;;
esac

mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
RUNNER_LOG="${RUN_DIR}/runner.log"
: > "${SERVER_LOG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1
setup_runtime_caches
setup_eval_output_dir
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
PYTHONPATH_EXTRA=""
if [[ -d "${REPO_ROOT}/third_party/human-eval" ]]; then
  PYTHONPATH_EXTRA="${REPO_ROOT}/third_party/human-eval"
fi
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:+${PYTHONPATH_EXTRA}:}${OSCAR_ROOT}/sglang-research/python"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

probe_ready() {
  if [[ "${API}" == "chat" ]]; then
    curl --noproxy '*' --max-time 10 -sf \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ready\"}],\"max_tokens\":1,\"temperature\":0}" \
      "http://127.0.0.1:${PORT}/v1/chat/completions" >/dev/null 2>&1
  else
    curl --noproxy '*' --max-time 10 -sf \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL}\",\"prompt\":\"ready\",\"max_tokens\":1,\"temperature\":0}" \
      "http://127.0.0.1:${PORT}/v1/completions" >/dev/null 2>&1
  fi
}

SERVER_ARGS=(
  --model-path "${MODEL}"
  --tensor-parallel-size 1
  --host 127.0.0.1
  --port "${PORT}"
  --dist-init-addr "127.0.0.1:${DIST_PORT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --kv-cache-dtype "${KV_DTYPE}"
  --prefill-attention-backend triton
  --decode-attention-backend triton
  --disable-piecewise-cuda-graph
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
  --page-size "${PAGE_SIZE}"
  --trust-remote-code
)
if [[ "${KV_DTYPE}" == "int2" ]]; then
  SERVER_ARGS+=(--kv-cache-quant-group-size 128)
fi

echo "[eval simple granite] task=${TASK} mode=${MODE} model=${MODEL} out=${RUN_DIR}"
env "${MODE_ENV[@]}" "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  if curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    echo "[eval simple granite] server api-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval simple granite] server died" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done
if ! curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
  echo "[eval simple granite] server not api-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi

for _ in $(seq 1 120); do
  if probe_ready; then
    echo "[eval simple granite] ${API}-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval simple granite] server died during ${API} probe" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
done
if ! probe_ready; then
  echo "[eval simple granite] server not ${API}-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi
sleep "${POST_READY_SLEEP}"

RUN_ARGS=(
  -m sglang.test.run_eval
  --base-url "http://127.0.0.1:${PORT}"
  --model "${MODEL_REL}"
  --eval-name "${TASK}"
  --api "${API}"
  --num-threads "${NUM_THREADS}"
  --max-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --repeat "${REPEAT}"
)
if [[ -n "${NUM_EXAMPLES}" ]]; then
  RUN_ARGS+=(--num-examples "${NUM_EXAMPLES}")
fi
if [[ "${TASK}" == "gsm8k" && -n "${GSM8K_DATA_PATH:-}" ]]; then
  RUN_ARGS+=(--gsm8k-data-path "${GSM8K_DATA_PATH}")
fi

"${PY}" "${RUN_ARGS[@]}" 2>&1 | tee "${RUNNER_LOG}"

MODEL_SLUG="${MODEL_REL//\//_}"
RESULT_JSON="${SGLANG_EVAL_OUTPUT_DIR}/${TASK}_${MODEL_SLUG}.json"
RESULT_HTML="${SGLANG_EVAL_OUTPUT_DIR}/${TASK}_${MODEL_SLUG}.html"
if [[ -f "${RESULT_JSON}" ]]; then
  cp "${RESULT_JSON}" "${RUN_DIR}/metrics.json"
fi
if [[ -f "${RESULT_HTML}" ]]; then
  cp "${RESULT_HTML}" "${RUN_DIR}/report.html"
fi
