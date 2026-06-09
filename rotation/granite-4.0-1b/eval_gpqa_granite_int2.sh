#!/usr/bin/env bash
# GPQA eval for Granite 4.0 1B with plain INT2 KV (no OSCAR rotation).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
# shellcheck source=../../scripts/lib/eval_cli.sh
source "${REPO_ROOT}/scripts/lib/eval_cli.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"

MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/GPQA/eval_int2_$(date +%Y%m%dT%H%M%S)}"
PY="${PY:-.venv-oscar-kv/bin/python}"
GPU="${GPU:-0}"
PORT="${PORT:-31150}"
DIST_PORT="${DIST_PORT:-41150}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-198}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
POST_READY_SLEEP="${POST_READY_SLEEP:-1}"

EVAL_CLI_HELP_FN=eval_cli_usage_gpqa
parse_eval_cli_args "$@"
MODEL="$(resolve_repo_path "${MODEL}")"
PY="$(resolve_repo_path "${PY}")"

if [[ ! -x "${PY}" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[eval granite int2] missing model config: ${MODEL}/config.json" >&2
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
  --page-size 128
  --trust-remote-code
)

echo "[eval granite int2] model=${MODEL} out=${RUN_DIR}"
"${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  if curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    echo "[eval granite int2] server api-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval granite int2] server died" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done
if ! curl --noproxy '*' --max-time 5 -sf "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
  echo "[eval granite int2] server not api-ready" >&2
  tail -100 "${SERVER_LOG}" || true
  exit 1
fi

for _ in $(seq 1 120); do
  if probe_completions_ready; then
    echo "[eval granite int2] completions-ready"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval granite int2] server died during completions probe" >&2
    tail -100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
done
if ! probe_completions_ready; then
  echo "[eval granite int2] server not completions-ready" >&2
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
