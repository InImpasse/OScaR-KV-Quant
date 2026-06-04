#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build/bin/llama-bench}"

MODEL="${MODEL:-$HOME/models/gguf/granite-4.0-1b-base-bf16.gguf}"
CONTEXT="${CONTEXT:-32768}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
GEN_TOKENS="${GEN_TOKENS:-512}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
KV_TYPES="${KV_TYPES:-f16,q8_0,q4_0,q2_0}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$RUNS_DIR/llamacpp_kv_$STAMP"

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
} > "$OUT_DIR/config.txt"

IFS=',' read -ra TYPES <<< "$KV_TYPES"
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
