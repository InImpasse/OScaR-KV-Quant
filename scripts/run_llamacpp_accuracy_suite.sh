#!/usr/bin/env bash
# Run llama.cpp/llama-server accuracy suite for BF16 and INT4 delivery variants.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-server}"
LLAMA_EVAL="${LLAMA_EVAL:-$ROOT_DIR/third_party/OSCAR/examples/llama-eval/llama-eval.py}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/llamacpp_accuracy_suite_$STAMP}"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
VARIANTS="${VARIANTS:-baseline_bf16,oscar_int4,plain_int4}"
DATASETS="${DATASETS:-gpqa,gsm8k,math500,humaneval,aime2025}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
PORT="${PORT:-8033}"
THREADS="${THREADS:-1}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-120}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-0}"
SEED="${SEED:-1234}"
TEMPERATURE="${TEMPERATURE:-0}"
DRY_RUN="${DRY_RUN:-1}"
ACK_EVAL="${ACK_EVAL:-0}"
ALLOW_HUMANEVAL_EXEC="${ALLOW_HUMANEVAL_EXEC:-0}"

GPQA_N_CASES="${GPQA_N_CASES:-198}"
GSM8K_N_CASES="${GSM8K_N_CASES:-200}"
MATH500_N_CASES="${MATH500_N_CASES:-500}"
HUMANEVAL_N_CASES="${HUMANEVAL_N_CASES:-164}"
AIME25_N_CASES="${AIME25_N_CASES:-60}"
GPQA_N_PREDICT="${GPQA_N_PREDICT:-96}"
GSM8K_N_PREDICT="${GSM8K_N_PREDICT:-128}"
MATH500_N_PREDICT="${MATH500_N_PREDICT:-2048}"
HUMANEVAL_N_PREDICT="${HUMANEVAL_N_PREDICT:-512}"
AIME25_N_PREDICT="${AIME25_N_PREDICT:-2048}"

usage() {
  cat <<'EOF'
Usage: scripts/run_llamacpp_accuracy_suite.sh [env overrides]

Environment:
  OUT_DIR=DIR
  VARIANTS=baseline_bf16,oscar_int4,plain_int4
  DATASETS=gpqa,gsm8k,math500,humaneval,aime2025
  *_N_CASES and *_N_PREDICT override per-dataset sizes.
  DRY_RUN=0 ACK_EVAL=1 executes.
  ALLOW_HUMANEVAL_EXEC=1 is required when DATASETS includes humaneval.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DRY_RUN" != "1" && "$ACK_EVAL" != "1" ]]; then
  echo "Refusing real eval without ACK_EVAL=1." >&2
  exit 2
fi
if [[ ",$DATASETS," == *",humaneval,"* && "$DRY_RUN" != "1" && "$ALLOW_HUMANEVAL_EXEC" != "1" ]]; then
  echo "Refusing HumanEval real run without ALLOW_HUMANEVAL_EXEC=1 because it executes generated code in the grader." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_SERVER" ]] || { echo "llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
  [[ -f "$BASE_MODEL" ]] || { echo "BASE_MODEL not found: $BASE_MODEL" >&2; exit 1; }
  [[ -f "$OSCAR_MODEL" ]] || { echo "OSCAR_MODEL not found: $OSCAR_MODEL" >&2; exit 1; }
  mkdir -p "$OUT_DIR/raw" "$OUT_DIR/logs"
fi

mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/config.txt" <<EOF
base_model=$BASE_MODEL
oscar_model=$OSCAR_MODEL
variants=$VARIANTS
datasets=$DATASETS
ctx_size=$CTX_SIZE
seed=$SEED
temperature=$TEMPERATURE
dry_run=$DRY_RUN
EOF

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

enabled() {
  local list="$1"
  local needle="$2"
  [[ ",$list," == *",$needle,"* || "$list" == "all" ]]
}

dataset_args() {
  case "$1" in
    gpqa) printf '%s %s\n' "$GPQA_N_CASES" "$GPQA_N_PREDICT" ;;
    gsm8k) printf '%s %s\n' "$GSM8K_N_CASES" "$GSM8K_N_PREDICT" ;;
    math500) printf '%s %s\n' "$MATH500_N_CASES" "$MATH500_N_PREDICT" ;;
    humaneval) printf '%s %s\n' "$HUMANEVAL_N_CASES" "$HUMANEVAL_N_PREDICT" ;;
    aime2025) printf '%s %s\n' "$AIME25_N_CASES" "$AIME25_N_PREDICT" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

run_variant() {
  local label="$1"
  local model="$2"
  local cache_k="$3"
  local cache_v="$4"
  local no_hadamard="$5"
  local clip_ratio="$6"
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
      LLAMA_KV_CLIP_RATIO_V="$clip_ratio"
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

  for dataset in gpqa gsm8k math500 humaneval aime2025; do
    enabled "$DATASETS" "$dataset" || continue
    read -r n_cases n_predict < <(dataset_args "$dataset")
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

enabled "$VARIANTS" baseline_bf16 && run_variant baseline_bf16 "$BASE_MODEL" bf16 bf16 0 0
enabled "$VARIANTS" oscar_int4 && run_variant oscar_int4 "$OSCAR_MODEL" q4_0 q4_0 1 0.96
enabled "$VARIANTS" plain_int4 && run_variant plain_int4 "$BASE_MODEL" q4_0 q4_0 0 0
enabled "$VARIANTS" oscar_int2 && run_variant oscar_int2 "$OSCAR_MODEL" q2_0 q2_0 1 0
enabled "$VARIANTS" plain_int2 && run_variant plain_int2 "$BASE_MODEL" q2_0 q2_0 0 0

if [[ "$DRY_RUN" != "1" ]]; then
  python3 "$ROOT_DIR/scripts/summarize_llamacpp_accuracy_suite.py" "$OUT_DIR"
  echo "Results written to: $OUT_DIR"
else
  echo "Dry run complete; no results written."
fi
