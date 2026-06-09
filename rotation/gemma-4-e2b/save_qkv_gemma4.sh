#!/usr/bin/env bash
# Dump Q/K/V for Gemma 4 E2B.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"
SGLANG_DUMP_DIR="${OSCAR_ROOT}/sglang-dump-qkv"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="${MODEL:-checkpoints/gemma-4-E2B}"
MODEL="$(resolve_repo_path "${MODEL}")"
setup_runtime_caches
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31120}"
DIST_PORT="${DIST_PORT:-41120}"
GPU="${GPU:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-8}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1800}"

export DUMP_KVCACHE="${DUMP_KVCACHE:-true}"
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-6000}"

DATASET="${DATASET:-GPQA}"
GROUP_SIZE="${GROUP_SIZE:-128}"
CALIB_DIR="${SCRIPT_DIR}/${DATASET}/latest"
export DUMP_KVCACHE_DIR="${DUMP_KVCACHE_DIR:-${CALIB_DIR}/qkv_dumps/gpqa}"
mkdir -p "${DUMP_KVCACHE_DIR}"

PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[save_qkv gemma4] missing model config: ${MODEL}/config.json" >&2
  echo "  Set MODEL=/path/to/gemma checkpoint or run scripts/download_models.sh" >&2
  exit 1
fi
if [[ ! -d "${SGLANG_DUMP_DIR}/python" ]]; then
  echo "[save_qkv gemma4] missing SGLang dump tree: ${SGLANG_DUMP_DIR}/python" >&2
  echo "  Run git submodule update --init --recursive" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[save_qkv gemma4] curl is required" >&2
  exit 1
fi
if ! "${PY}" -m sglang.launch_server --help >/dev/null 2>&1; then
  echo "[save_qkv gemma4] python cannot run sglang.launch_server; run scripts/install_sglang_os.sh" >&2
  exit 1
fi

LOCAL_PYTHONPATH="${OSCAR_ROOT}/rotation/_dump_compat:${SGLANG_DUMP_DIR}/python"
if [[ -n "${PYTHONPATH:-}" ]]; then
  LOCAL_PYTHONPATH="${LOCAL_PYTHONPATH}:${PYTHONPATH}"
fi
export PYTHONPATH="${LOCAL_PYTHONPATH}"

SERVER_LOG="${DUMP_KVCACHE_DIR}/server.log"
DUMP_RUNNER_LOG="${DUMP_KVCACHE_DIR}/dump_runner.log"
: > "${SERVER_LOG}"

log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVER_ARGS=(
  --model-path "${MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-queued-requests "${MAX_QUEUED_REQUESTS}"
  --page-size 128
  --chunked-prefill-size 2048
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --kv-cache-dtype auto
  --prefill-attention-backend triton
  --decode-attention-backend triton
  --host 127.0.0.1
  --port "${PORT}"
  --dist-init-addr "127.0.0.1:${DIST_PORT}"
  --trust-remote-code
  --disable-custom-all-reduce
  --disable-cuda-graph
  --watchdog-timeout 1800
)
if [[ -n "${REASONING_PARSER:-}" ]]; then
  SERVER_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

log "Starting sglang server for QKV dump (Gemma4 E2B)"
log "  model=${MODEL}"

PYTHONPATH="${LOCAL_PYTHONPATH}" \
  "${PY}" -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
  if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "Server ready after ${elapsed}s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

"${PY}" "${OSCAR_ROOT}/rotation/_eval_runner/dump_gpqa_prompts.py" \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --num-prompts 24 \
  --num-threads "${NUM_WORKERS:-4}" \
  --temperature 0.6 --top-p 0.95 --top-k 40 \
  --max-tokens 1 \
  2>&1 | tee "${DUMP_RUNNER_LOG}"

if [[ -d "${DUMP_KVCACHE_DIR}/layer_0/q" ]]; then
  N_PROMPTS=$("${PY}" - "${DUMP_KVCACHE_DIR}/layer_0/seq_lens" <<'PYEOF'
import os, sys, torch
seq_dir = sys.argv[1]
total = 0
for f in sorted(os.listdir(seq_dir), key=lambda x: int(x.split('.')[0])):
    s = torch.load(os.path.join(seq_dir, f), weights_only=True, map_location='cpu')
    total += len(s.tolist())
print(total)
PYEOF
  )
  FINAL_TAG="seq${DUMP_KVCACHE_TOKENS}_prompt${N_PROMPTS}_group${GROUP_SIZE}"
  FINAL_DIR="${SCRIPT_DIR}/${DATASET}/${FINAL_TAG}"
  rm -rf "${FINAL_DIR}"
  mv "${CALIB_DIR}" "${FINAL_DIR}"
  log "final_dir=${FINAL_DIR}"
fi
