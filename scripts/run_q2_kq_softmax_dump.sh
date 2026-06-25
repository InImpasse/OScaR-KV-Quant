#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-debug}"
MODEL="${MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/q2_kq_softmax_dump_current}"
PROMPT="${PROMPT:-Question: What is 2 + 2? Answer with one number.}"
CTX="${CTX:-128}"
PREDICT="${PREDICT:-1}"
DRY_RUN="${DRY_RUN:-1}"
ACK_Q2_KQ_DUMP="${ACK_Q2_KQ_DUMP:-0}"

if [[ "$DRY_RUN" != "1" && "$ACK_Q2_KQ_DUMP" != "1" ]]; then
  echo "Refusing runtime KQ dump without ACK_Q2_KQ_DUMP=1. Use DRY_RUN=1 to inspect commands." >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "missing llama-debug binary: $BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

run_case() {
  local name="$1"
  local cache_k="$2"
  local cache_v="$3"
  local owht="$4"
  local nohad="$5"
  local clip="$6"

  local case_dir="$OUT_DIR/$name"
  local tensor_dir="$case_dir/tensors"
  local -a envs=(
    "LLAMA_DEBUG_TENSOR_DUMP_DIR=$tensor_dir"
    "LLAMA_DEBUG_TENSOR_DUMP_ONLY=1"
    "LLAMA_KV_Q2_0_OWHT=$owht"
    "LLAMA_KV_NO_HADAMARD=$nohad"
    "LLAMA_KV_CLIP_RATIO=$clip"
  )
  local -a cmd=(
    "$BIN"
    -m "$MODEL"
    -p "$PROMPT"
    -n "$PREDICT"
    -c "$CTX"
    -ngl 999
    -fa off
    --cache-type-k "$cache_k"
    --cache-type-v "$cache_v"
    --no-warmup
    --log-disable
    --verbosity 0
    --tensor-filter 'kq($|-)'
    --tensor-filter 'kq_soft_max'
    --tensor-filter 'kqv_out'
    --tensor-filter 'Qcur'
    --tensor-filter 'Kcur'
    --tensor-filter 'cache_k_set_rows'
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s:' "$name"
    printf ' %q' "${envs[@]}" "${cmd[@]}"
    printf '\n'
    return
  fi

  rm -rf "$case_dir"
  mkdir -p "$tensor_dir"
  env "${envs[@]}" "${cmd[@]}" > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"
}

run_case oscar_bf16      bf16 bf16 0 0 0
run_case oscar_kq4_vbf16 q4_0 bf16 0 0 0
run_case oscar_kq2_vbf16 q2_0 bf16 1 1 0
