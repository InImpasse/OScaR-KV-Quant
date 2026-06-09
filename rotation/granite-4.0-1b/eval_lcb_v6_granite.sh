#!/usr/bin/env bash
# LiveCodeBench v6 code-generation eval for Granite 4.0 1B across BF16,
# plain INT2, and OSCAR INT2 KV modes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"
LCB_ROOT="${LIVE_CODE_BENCH_ROOT:-third_party/LiveCodeBench}"
LCB_ROOT="$(resolve_repo_path "${LCB_ROOT}")"

MODE="${MODE:-bf16}"  # bf16 | int2 | oscar-int2
MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
MODEL="$(resolve_repo_path "${MODEL}")"
LCB_MODEL_NAME="${LCB_MODEL_NAME:-granite-4.0-1b-base}"
ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/GPQA/seq30000_prompt118_group128/rotations}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/LCB_V6/eval_${MODE}_$(date +%Y%m%dT%H%M%S)}"
PY="${PY:-.venv-oscar-kv/bin/python}"
PY="$(resolve_repo_path "${PY}")"
GPU="${GPU:-0}"
PORT="${PORT:-32400}"
DIST_PORT="${DIST_PORT:-42400}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-4}"
POST_READY_SLEEP="${POST_READY_SLEEP:-3}"
LCB_N="${LCB_N:-1}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0.2}"
LCB_TOP_P="${LCB_TOP_P:-0.95}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-2000}"
LCB_RELEASE="${LCB_RELEASE:-release_v6}"
LCB_MULTIPROCESS="${LCB_MULTIPROCESS:-1}"
LCB_NUM_PROCESS_EVALUATE="${LCB_NUM_PROCESS_EVALUATE:-1}"
LCB_TIMEOUT="${LCB_TIMEOUT:-6}"
LCB_EXTRA_ARGS=(${LCB_EXTRA_ARGS:-})

if [[ ! -d "${LCB_ROOT}/lcb_runner" ]]; then
  cat >&2 <<EOF
[eval lcb granite] missing LiveCodeBench checkout: ${LCB_ROOT}

Set up the official harness first:

  git clone https://github.com/LiveCodeBench/LiveCodeBench.git third_party/LiveCodeBench

This wrapper uses the checkout directly through PYTHONPATH.
EOF
  exit 2
fi
if [[ ! -x "${PY}" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[eval lcb granite] missing model config: ${MODEL}/config.json" >&2
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
    K_ROT="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}"
    V_ROT="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}"
    if [[ ! -f "${K_ROT}" || ! -f "${V_ROT}" ]]; then
      echo "[eval lcb granite] missing rotation files under ROT_DIR=${ROT_DIR}" >&2
      exit 1
    fi
    MODE_ENV=(
      SGLANG_ENABLE_MIXED_KV_WINDOWS=1
      SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
      SGLANG_OSCAR_ABSORB_V_ROTATION=1
      SGLANG_MIXED_KV_HP_MAX_SPLITS="${SGLANG_MIXED_KV_HP_MAX_SPLITS:-2}"
      SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
      SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
      SGLANG_MIXED_KV_MAX_QUANT_TOKENS="${SGLANG_MIXED_KV_MAX_QUANT_TOKENS:-32768}"
      SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS:-1024}"
      SGLANG_MIXED_KV_HP_DTYPE=bfloat16
      SGLANG_MIXED_KV_SCALE_DTYPE="${SGLANG_MIXED_KV_SCALE_DTYPE:-bfloat16}"
      SGLANG_OSCAR_K_ROTATION_PATH="${K_ROT}"
      SGLANG_OSCAR_V_ROTATION_PATH="${V_ROT}"
      SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-0.96}"
      SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-0.92}"
      SGLANG_LLOYD_MAX="${SGLANG_LLOYD_MAX:-0}"
      SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT="${SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT:-1}"
    )
    ;;
  *)
    echo "[eval lcb granite] unknown MODE=${MODE}; use bf16, int2, or oscar-int2" >&2
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
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONPATH="${LCB_ROOT}:${OSCAR_ROOT}/sglang-research/python:${PYTHONPATH:-}"
export OPENAI_KEY="${OPENAI_KEY:-EMPTY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LCB_USE_COMPLETIONS="${LCB_USE_COMPLETIONS:-1}"
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
  curl --noproxy '*' --max-time 10 -sf \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"ready\",\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:${PORT}/v1/completions" >/dev/null 2>&1
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

echo "[eval lcb granite] mode=${MODE} release=${LCB_RELEASE} model=${MODEL} out=${RUN_DIR}"
env "${MODE_ENV[@]}" "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  if curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    echo "[eval lcb granite] server api-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval lcb granite] server died" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done
if ! curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
  echo "[eval lcb granite] server not api-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi

for _ in $(seq 1 120); do
  if probe_ready; then
    echo "[eval lcb granite] completion-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval lcb granite] server died during completion probe" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
done
if ! probe_ready; then
  echo "[eval lcb granite] server not completion-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi
sleep "${POST_READY_SLEEP}"

(
  cd "${LCB_ROOT}"
  "${PY}" -m lcb_runner.runner.main \
    --model "${LCB_MODEL_NAME}" \
    --scenario codegeneration \
    --release_version "${LCB_RELEASE}" \
    --evaluate \
    --n "${LCB_N}" \
    --temperature "${LCB_TEMPERATURE}" \
    --top_p "${LCB_TOP_P}" \
    --max_tokens "${LCB_MAX_TOKENS}" \
    --multiprocess "${LCB_MULTIPROCESS}" \
    --num_process_evaluate "${LCB_NUM_PROCESS_EVALUATE}" \
    --timeout "${LCB_TIMEOUT}" \
    "${LCB_EXTRA_ARGS[@]}"
) 2>&1 | tee "${RUNNER_LOG}"

mkdir -p "${RUN_DIR}/lcb_output"
if [[ -d "${LCB_ROOT}/output/Granite-4.0-1B-Base" ]]; then
  cp -a "${LCB_ROOT}/output/Granite-4.0-1B-Base/." "${RUN_DIR}/lcb_output/"
fi
