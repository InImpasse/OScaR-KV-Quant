#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"

MODEL="${MODEL:-$HOME/models/gguf/granite-4.0-1b-base-bf16.gguf}"
CONTEXT="${CONTEXT:-32768}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
GEN_TOKENS="${GEN_TOKENS:-512}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
KV_TYPES="${KV_TYPES:-f16,q8_0,q4_0,q2_0}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
DRY_RUN="${DRY_RUN:-1}"
ACK_HEAVY_CONTEXT="${ACK_HEAVY_CONTEXT:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$RUNS_DIR/llamacpp_kv_$STAMP"

if [[ "$DRY_RUN" != "1" ]] && {
    (( CONTEXT >= 32768 )) || (( PROMPT_TOKENS >= 4096 )) || (( GEN_TOKENS >= 512 ));
  } && [[ "$ACK_HEAVY_CONTEXT" != "1" ]]; then
  echo "Refusing heavy KV benchmark without ACK_HEAVY_CONTEXT=1." >&2
  echo "Current settings: CONTEXT=$CONTEXT PROMPT_TOKENS=$PROMPT_TOKENS GEN_TOKENS=$GEN_TOKENS" >&2
  echo "Use DRY_RUN=1 to inspect commands or ACK_HEAVY_CONTEXT=1 DRY_RUN=0 to run intentionally." >&2
  exit 1
fi

IFS=',' read -ra TYPES <<< "$KV_TYPES"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN config:"
  echo "  model=$MODEL"
  echo "  context=$CONTEXT"
  echo "  prompt_tokens=$PROMPT_TOKENS"
  echo "  gen_tokens=$GEN_TOKENS"
  echo "  n_gpu_layers=$N_GPU_LAYERS"
  echo "  kv_types=$KV_TYPES"
  echo "  output_dir=$OUT_DIR"
  echo
  for kv in "${TYPES[@]}"; do
    printf 'DRY_RUN command:'
    printf ' %q' "$LLAMA_BENCH" \
      -m "$MODEL" \
      -c "$CONTEXT" \
      -p "$PROMPT_TOKENS" \
      -n "$GEN_TOKENS" \
      -ngl "$N_GPU_LAYERS" \
      --cache-type-k "$kv" \
      --cache-type-v "$kv" \
      --output json
    printf ' > %q\n' "$OUT_DIR/$kv.json"
  done
  echo
  echo "Dry run complete; no results written. Set DRY_RUN=0 ACK_HEAVY_CONTEXT=1 to run heavy settings intentionally."
  exit 0
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
  echo "llama-bench not found at $LLAMA_BENCH. Run ./scripts/build_llamacpp.sh first." >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Run ./scripts/download_gguf_models.sh or set MODEL=/path/to/model.gguf." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

{
  echo "model=$MODEL"
  echo "context=$CONTEXT"
  echo "prompt_tokens=$PROMPT_TOKENS"
  echo "gen_tokens=$GEN_TOKENS"
  echo "n_gpu_layers=$N_GPU_LAYERS"
  echo "kv_types=$KV_TYPES"
  echo "dry_run=$DRY_RUN"
  echo "ack_heavy_context=$ACK_HEAVY_CONTEXT"
} > "$OUT_DIR/config.txt"

for kv in "${TYPES[@]}"; do
  echo "Running KV cache type: $kv"
  "$LLAMA_BENCH" \
    -m "$MODEL" \
    -c "$CONTEXT" \
    -p "$PROMPT_TOKENS" \
    -n "$GEN_TOKENS" \
    -ngl "$N_GPU_LAYERS" \
    --cache-type-k "$kv" \
    --cache-type-v "$kv" \
    --output json \
    > "$OUT_DIR/$kv.json"
done

echo
echo "Results written to: $OUT_DIR"
