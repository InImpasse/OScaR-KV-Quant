#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-server}"
LLAMA_EVAL="${LLAMA_EVAL:-$ROOT_DIR/third_party/OSCAR/examples/llama-eval/llama-eval.py}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/gpqa_gsm8k_kv_eval_$STAMP}"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
CASES="${CASES:-baseline_bf16,oscar_turbo2_streamk,turbo2_streamk,oscar_turbo3,plain_int3,oscar_int4,plain_int4}"
DATASETS="${DATASETS:-gpqa,gsm8k}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
PORT="${PORT:-8033}"
THREADS="${THREADS:-1}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-120}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-0}"
GPQA_N_CASES="${GPQA_N_CASES:-198}"
GSM8K_N_CASES="${GSM8K_N_CASES:-100}"
GPQA_N_PREDICT="${GPQA_N_PREDICT:-96}"
GSM8K_N_PREDICT="${GSM8K_N_PREDICT:-128}"
SEED="${SEED:-1234}"
TEMPERATURE="${TEMPERATURE:-0}"
DRY_RUN="${DRY_RUN:-1}"
ACK_EVAL="${ACK_EVAL:-0}"

if [[ "$DRY_RUN" != "1" && "$ACK_EVAL" != "1" ]]; then
  echo "Refusing real eval without ACK_EVAL=1." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_SERVER" ]] || { echo "llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
  [[ -f "$BASE_MODEL" ]] || { echo "BASE_MODEL not found: $BASE_MODEL" >&2; exit 1; }
  [[ -f "$OSCAR_MODEL" ]] || { echo "OSCAR_MODEL not found: $OSCAR_MODEL" >&2; exit 1; }
  mkdir -p "$OUT_DIR/raw" "$OUT_DIR/logs"
  cat > "$OUT_DIR/config.txt" <<EOF
llama_server=$LLAMA_SERVER
llama_eval=$LLAMA_EVAL
base_model=$BASE_MODEL
oscar_model=$OSCAR_MODEL
cases=$CASES
datasets=$DATASETS
ctx_size=$CTX_SIZE
ctx_note=accuracy eval defaults to 4096 because GPQA/GSM8K prompts fit comfortably; 32k KV speed/memory is measured by bench_32k_llamacpp_kv.sh.
gpqa_n_cases=$GPQA_N_CASES
gsm8k_n_cases=$GSM8K_N_CASES
gpqa_n_predict=$GPQA_N_PREDICT
gsm8k_n_predict=$GSM8K_N_PREDICT
seed=$SEED
temperature=$TEMPERATURE
EOF
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/health"
  local deadline=$(( $(date +%s) + SERVER_TIMEOUT_SEC ))
  while true; do
    if NO_PROXY='*' no_proxy='*' python3 - "$url" >/dev/null 2>&1 <<'PY'
import sys
import requests
s = requests.Session()
s.trust_env = False
try:
    r = s.get(sys.argv[1], timeout=2)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if r.status_code == 200 else 1)
PY
    then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "server did not become ready: $url" >&2
      return 1
    fi
    sleep 1
  done
}

case_enabled() {
  local needle="$1"
  [[ ",$CASES," == *",$needle,"* || "$CASES" == "all" ]]
}

dataset_enabled() {
  local needle="$1"
  [[ ",$DATASETS," == *",$needle,"* || "$DATASETS" == "all" ]]
}

run_variant() {
  local label="$1"
  local model="$2"
  local cache_k="$3"
  local cache_v="$4"
  local no_hadamard="$5"
  local clip_ratio="$6"
  local turbo_stream_k="${7:-0}"
  local q2_owht="0"
  if [[ "$cache_k/$cache_v" == "q2_0/q2_0" && "$no_hadamard" == "1" ]]; then
    q2_owht="1"
  fi

  local server_cmd=(
    env
      LLAMA_KV_HP_SINK=0
      LLAMA_KV_HP_RECENT=0
      LLAMA_KV_Q2_0_OWHT="$q2_owht"
      LLAMA_KV_NO_HADAMARD="$no_hadamard"
      LLAMA_KV_CLIP_RATIO="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_K="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_V="$([[ "$no_hadamard" == "1" && "$clip_ratio" == "0.96" ]] && printf '0.92' || printf '%s' "$clip_ratio")"
      LLAMA_TURBO_VEC_STREAM_K="$turbo_stream_k"
    "$LLAMA_SERVER"
      -m "$model"
      -c "$CTX_SIZE"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      -ctk "$cache_k"
      -ctv "$cache_v"
      --host 127.0.0.1
      --port "$PORT"
      --no-webui
      --chat-template granite
      -np 1
      --cache-ram 0
      --no-cache-prompt
      --ctx-checkpoints 0
      --no-warmup
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN server %s:' "$label"
    printf ' %q' "${server_cmd[@]}"
    printf '\n'
  else
    echo "Starting $label"
    "${server_cmd[@]}" > "$OUT_DIR/logs/${label}.server.log" 2>&1 &
    server_pid="$!"
    wait_for_server
  fi

  for dataset in gpqa gsm8k; do
    dataset_enabled "$dataset" || continue
    local n_cases n_predict
    if [[ "$dataset" == "gpqa" ]]; then
      n_cases="$GPQA_N_CASES"
      n_predict="$GPQA_N_PREDICT"
    else
      n_cases="$GSM8K_N_CASES"
      n_predict="$GSM8K_N_PREDICT"
    fi

    local eval_cmd=(
      python3 "$LLAMA_EVAL"
        --server "http://127.0.0.1:${PORT}"
        --server-name "$label"
        --dataset "$dataset"
        --n_cases "$n_cases"
        --seed "$SEED"
        --n_predict "$n_predict"
        --temperature "$TEMPERATURE"
        --threads "$THREADS"
        --model llama
        --grader-type regex
        --output "$OUT_DIR/raw/${label}_${dataset}.json"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY_RUN eval %s/%s:' "$label" "$dataset"
      printf ' %q' "${eval_cmd[@]}"
      printf '\n'
    else
      echo "Running $label $dataset"
      if command -v timeout >/dev/null 2>&1 && (( EVAL_TIMEOUT_SEC > 0 )); then
        timeout --signal=INT --kill-after=30s "${EVAL_TIMEOUT_SEC}s" "${eval_cmd[@]}" \
          > "$OUT_DIR/logs/${label}_${dataset}.eval.log" 2>&1
      else
        "${eval_cmd[@]}" > "$OUT_DIR/logs/${label}_${dataset}.eval.log" 2>&1
      fi
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    cleanup
    server_pid=""
  fi
}

case_enabled baseline_bf16 && run_variant baseline_bf16 "$BASE_MODEL" bf16 bf16 0 0 0
case_enabled oscar_turbo2_streamk && run_variant oscar_turbo2_streamk "$OSCAR_MODEL" turbo2 turbo2 1 0.96 1
case_enabled turbo2_streamk && run_variant turbo2_streamk "$BASE_MODEL" turbo2 turbo2 0 0 1
case_enabled oscar_turbo3 && run_variant oscar_turbo3 "$OSCAR_MODEL" turbo3 turbo3 1 0.96 0
case_enabled plain_int3 && run_variant plain_int3 "$BASE_MODEL" turbo3 turbo3 0 0 0
case_enabled oscar_int4 && run_variant oscar_int4 "$OSCAR_MODEL" q4_0 q4_0 1 0 0
case_enabled plain_int4 && run_variant plain_int4 "$BASE_MODEL" q4_0 q4_0 0 0 0

if [[ "$DRY_RUN" != "1" ]]; then
  python3 "$ROOT_DIR/scripts/summarize_gpqa_gsm8k_kv_eval.py" "$OUT_DIR"
  echo "Results written to: $OUT_DIR"
else
  echo "Dry run complete; no results written."
fi
