#!/usr/bin/env bash
# Stable GPQA eval driver for Granite 4.0 1B base served via /v1/completions.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
# shellcheck source=../../scripts/lib/eval_cli.sh
source "${REPO_ROOT}/scripts/lib/eval_cli.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"

MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
ROT_DIR="${ROT_DIR:-${SCRIPT_DIR}/GPQA/seq30000_prompt118_group128/rotations}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/GPQA/eval_oscar_$(date +%Y%m%dT%H%M%S)}"
PY="${PY:-.venv-oscar-kv/bin/python}"
GPU="${GPU:-0}"
PORT="${PORT:-31120}"
DIST_PORT="${DIST_PORT:-41120}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-198}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
POST_READY_SLEEP="${POST_READY_SLEEP:-10}"
K_ROT_FILENAME="k_rotation_qqt_r_h_pbr.pt"
V_ROT_FILENAME="v_rotation_sst_r_h_pbr.pt"
MIXED_KV_HP_MAX_SPLITS="2"
MIXED_KV_PREFIX_TOKENS="64"
MIXED_KV_RECENT_TOKENS="256"
MIXED_KV_MAX_QUANT_TOKENS="32768"
MIXED_KV_SCALE_DTYPE="bfloat16"
OSCAR_K_CLIP_RATIO="0.96"
OSCAR_V_CLIP_RATIO="0.92"
LLOYD_MAX="0"
OSCAR_FUSED_ROTATE_CLIP_QUANT="1"

EVAL_CLI_HELP_FN=eval_cli_usage_gpqa
parse_eval_cli_args "$@"
MODEL="$(resolve_repo_path "${MODEL}")"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
PY="$(resolve_repo_path "${PY}")"

if [[ ! -x "${PY}" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[eval granite] missing model config: ${MODEL}/config.json" >&2
  exit 1
fi

K_ROT="${ROT_DIR}/${K_ROT_FILENAME}"
V_ROT="${ROT_DIR}/${V_ROT_FILENAME}"
if [[ ! -f "${K_ROT}" || ! -f "${V_ROT}" ]]; then
  echo "[eval granite] missing rotation files under ROT_DIR=${ROT_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
RUNNER_LOG="${RUN_DIR}/runner.log"
: > "${SERVER_LOG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1
setup_runtime_caches
export PYTHONPATH="${OSCAR_ROOT}/sglang-research/python:${PYTHONPATH:-}"
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

probe_completions_ready() {
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
  --kv-cache-dtype int2
  --kv-cache-quant-group-size 128
  --prefill-attention-backend triton
  --decode-attention-backend triton
  --disable-piecewise-cuda-graph
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
  --page-size 8
  --trust-remote-code
)

echo "[eval granite] model=${MODEL} rot=${ROT_DIR} out=${RUN_DIR}"
env \
  SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_OSCAR_ABSORB_V_ROTATION=1 \
  SGLANG_MIXED_KV_HP_MAX_SPLITS="${MIXED_KV_HP_MAX_SPLITS}" \
  SGLANG_MIXED_KV_PREFIX_TOKENS="${MIXED_KV_PREFIX_TOKENS}" \
  SGLANG_MIXED_KV_RECENT_TOKENS="${MIXED_KV_RECENT_TOKENS}" \
  SGLANG_MIXED_KV_MAX_QUANT_TOKENS="${MIXED_KV_MAX_QUANT_TOKENS}" \
  SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
  SGLANG_MIXED_KV_SCALE_DTYPE="${MIXED_KV_SCALE_DTYPE}" \
  SGLANG_OSCAR_K_ROTATION_PATH="${K_ROT}" \
  SGLANG_OSCAR_V_ROTATION_PATH="${V_ROT}" \
  SGLANG_OSCAR_K_CLIP_RATIO="${OSCAR_K_CLIP_RATIO}" \
  SGLANG_OSCAR_V_CLIP_RATIO="${OSCAR_V_CLIP_RATIO}" \
  SGLANG_LLOYD_MAX="${LLOYD_MAX}" \
  SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT="${OSCAR_FUSED_ROTATE_CLIP_QUANT}" \
  "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  if curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    echo "[eval granite] server api-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval granite] server died" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done
if ! curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
  echo "[eval granite] server not api-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi

for _ in $(seq 1 120); do
  if probe_completions_ready; then
    echo "[eval granite] completions-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval granite] server died during completions probe" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
done
if ! probe_completions_ready; then
  echo "[eval granite] server not completions-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi
sleep "${POST_READY_SLEEP}"

"${PY}" "${OSCAR_ROOT}/rotation/_eval_runner/run_gpqa_completions_eval.py" \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --num-examples "${NUM_EXAMPLES}" \
  --max-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --top-k "${TOP_K:-1}" \
  --max-retries "${MAX_RETRIES:-2}" \
  --timeout "${REQUEST_TIMEOUT:-600}" \
  --sleep-between "${SLEEP_BETWEEN:-0.25}" \
  --output-dir "${RUN_DIR}" \
  2>&1 | tee "${RUNNER_LOG}"
