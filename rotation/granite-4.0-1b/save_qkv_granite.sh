#!/usr/bin/env bash
# Dump post-RoPE Q/K/V tensors for Granite 4.0 1B OSCAR rotation calibration.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"
SGLANG_DUMP_DIR="${OSCAR_ROOT}/sglang-dump-qkv"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
MODEL="$(resolve_repo_path "${MODEL}")"
MODEL_REL="$(repo_relative_path "${MODEL}")"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31110}"
DIST_PORT="${DIST_PORT:-41110}"
GPU="${GPU:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-4}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-1800}"
POST_READY_SLEEP="${POST_READY_SLEEP:-10}"
CALIB_PROFILE="paper"

usage() {
  cat <<'EOF'
Usage: rotation/granite-4.0-1b/save_qkv_granite.sh [options]

Options:
  --calib-profile PROFILE  paper | smoke (default: paper)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --calib-profile) CALIB_PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export DUMP_KVCACHE="${DUMP_KVCACHE:-true}"
case "${CALIB_PROFILE}" in
  paper)
    DEFAULT_DUMP_TOKENS=30000
    DEFAULT_NUM_PROMPTS=198
    DEFAULT_NUM_WORKERS=1
    ;;
  smoke)
    DEFAULT_DUMP_TOKENS=2000
    DEFAULT_NUM_PROMPTS=32
    DEFAULT_NUM_WORKERS=1
    ;;
  *)
    echo "[save_qkv granite] unknown CALIB_PROFILE=${CALIB_PROFILE}; use paper or smoke" >&2
    exit 1
    ;;
esac
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-${DEFAULT_DUMP_TOKENS}}"

DATASET="${DATASET:-GPQA}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DUMP_API="${DUMP_API:-completions}"
NUM_PROMPTS="${NUM_PROMPTS:-${DEFAULT_NUM_PROMPTS}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
MIN_DUMPED_TOKENS="${MIN_DUMPED_TOKENS:-$((DUMP_KVCACHE_TOKENS * 9 / 10))}"
MAX_DUMP_ERRORS="${MAX_DUMP_ERRORS:-0}"
MIN_SUCCESS_PROMPTS="${MIN_SUCCESS_PROMPTS:-1}"
CALIB_DIR="${SCRIPT_DIR}/${DATASET}/latest"
export DUMP_KVCACHE_DIR="${DUMP_KVCACHE_DIR:-${CALIB_DIR}/qkv_dumps/gpqa}"
mkdir -p "${DUMP_KVCACHE_DIR}"

PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1
setup_runtime_caches

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[save_qkv granite] missing model config: ${MODEL}/config.json" >&2
  echo "  Set MODEL=/path/to/granite checkpoint or run scripts/download_models.sh" >&2
  exit 1
fi
if [[ ! -d "${SGLANG_DUMP_DIR}/python" ]]; then
  echo "[save_qkv granite] missing SGLang dump tree: ${SGLANG_DUMP_DIR}/python" >&2
  echo "  Run git submodule update --init --recursive" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[save_qkv granite] curl is required" >&2
  exit 1
fi
if ! "${PY}" -m sglang.launch_server --help >/dev/null 2>&1; then
  echo "[save_qkv granite] python cannot run sglang.launch_server; run scripts/install_sglang_os.sh" >&2
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
    log "Stopping server PID ${SERVER_PID}"
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

log "Starting sglang server for QKV dump (Granite)"
log "  oscar_root=${OSCAR_ROOT}"
log "  model=${MODEL}"
log "  calib_profile=${CALIB_PROFILE}"
log "  dump_tokens=${DUMP_KVCACHE_TOKENS}"
log "  num_prompts=${NUM_PROMPTS} num_workers=${NUM_WORKERS}"
log "  dump_api=${DUMP_API}"
log "  post_ready_sleep=${POST_READY_SLEEP}s"

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
    log "Server died. Last log:"
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

elapsed=0
while [[ "${elapsed}" -lt "${MAX_WAIT_SECS}" ]]; do
  if grep -qE "Application startup complete|Uvicorn running" "${SERVER_LOG}"; then
    log "Server application ready after health check"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Server died after health check. Last log:"
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done
if ! grep -qE "Application startup complete|Uvicorn running" "${SERVER_LOG}"; then
  log "Server app not ready after ${MAX_WAIT_SECS}s"
  tail -80 "${SERVER_LOG}" || true
  exit 1
fi

if [[ "${POST_READY_SLEEP}" != "0" ]]; then
  log "Sleeping ${POST_READY_SLEEP}s after app startup before sending prompts"
  sleep "${POST_READY_SLEEP}"
fi

log "Sending GPQA prompts via dump runner"
"${PY}" "${OSCAR_ROOT}/rotation/_eval_runner/dump_gpqa_prompts.py" \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --num-prompts "${NUM_PROMPTS}" \
  --num-threads "${NUM_WORKERS}" \
  --temperature 0.6 --top-p 0.95 --top-k 40 \
  --max-tokens 1 \
  --api "${DUMP_API}" \
  --dump-dir "${DUMP_KVCACHE_DIR}" \
  --dump-token-budget "${DUMP_KVCACHE_TOKENS}" \
  2>&1 | tee "${DUMP_RUNNER_LOG}"

read_dump_stat() {
  local key="$1"
  "${PY}" - "${DUMP_RUNNER_LOG}" "${key}" <<'PYEOF'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
key = sys.argv[2]
m = re.search(r"\[dump\] done .*? submitted=(\d+) ok=(\d+)\s+err=(\d+)\s+dumped_tokens=(\d+)", text)
if not m:
    raise SystemExit(f"missing dump summary in {sys.argv[1]}")
stats = dict(zip(("submitted", "ok", "err", "dumped_tokens"), map(int, m.groups())))
print(stats[key])
PYEOF
}

if [[ ! -d "${DUMP_KVCACHE_DIR}/layer_0/q" ]]; then
  log "QKV dump did not create ${DUMP_KVCACHE_DIR}/layer_0/q"
  tail -80 "${DUMP_RUNNER_LOG}" || true
  exit 1
fi

SUBMITTED="$(read_dump_stat submitted)"
OK_PROMPTS="$(read_dump_stat ok)"
ERR_PROMPTS="$(read_dump_stat err)"
DUMPED_TOKENS="$(read_dump_stat dumped_tokens)"
log "dump_summary submitted=${SUBMITTED} ok=${OK_PROMPTS} err=${ERR_PROMPTS} dumped_tokens=${DUMPED_TOKENS}"

if (( OK_PROMPTS < MIN_SUCCESS_PROMPTS )); then
  log "Only ${OK_PROMPTS} successful prompts; require >= ${MIN_SUCCESS_PROMPTS}"
  exit 1
fi
if (( ERR_PROMPTS > MAX_DUMP_ERRORS )); then
  log "Dump saw ${ERR_PROMPTS} request errors; require <= ${MAX_DUMP_ERRORS}"
  exit 1
fi
if (( DUMPED_TOKENS < MIN_DUMPED_TOKENS )); then
  log "Dumped ${DUMPED_TOKENS} tokens; require >= ${MIN_DUMPED_TOKENS}"
  exit 1
fi

# Rename calib dir (same logic as upstream qwen3-8B script) after strict checks.
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
  META_PATH="${CALIB_DIR}/calibration_meta.json"
  "${PY}" - "${META_PATH}" <<PYEOF
import json, time
meta = {
    "model": "${MODEL_REL}",
    "dataset": "${DATASET}",
    "calib_profile": "${CALIB_PROFILE}",
    "method": "qqt_sst",
    "composition": "r_h_pbr",
    "group_size": int("${GROUP_SIZE}"),
    "dump_token_budget": int("${DUMP_KVCACHE_TOKENS}"),
    "dumped_tokens": int("${DUMPED_TOKENS}"),
    "num_prompts_requested": int("${NUM_PROMPTS}"),
    "num_prompts_captured": int("${N_PROMPTS}"),
    "runner_ok_prompts": int("${OK_PROMPTS}"),
    "runner_error_prompts": int("${ERR_PROMPTS}"),
    "runner_submitted_prompts": int("${SUBMITTED}"),
    "min_dumped_tokens": int("${MIN_DUMPED_TOKENS}"),
    "max_dump_errors": int("${MAX_DUMP_ERRORS}"),
    "dump_api": "${DUMP_API}",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
with open("${META_PATH}", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PYEOF
  rm -rf "${FINAL_DIR}"
  mv "${CALIB_DIR}" "${FINAL_DIR}"
  log "final_dir=${FINAL_DIR}"
fi
