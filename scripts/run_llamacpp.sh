#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$ROOT_DIR/third_party/OSCAR/build/bin/llama-cli}"

MODEL="${MODEL:-$HOME/models/gguf/granite-4.0-1b-base-bf16.gguf}"
KV_TYPE="${KV_TYPE:-f16}"
CONTEXT="${CONTEXT:-32768}"
PREDICT="${PREDICT:-256}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
PROMPT="${PROMPT:-Explain KV cache memory in one concise paragraph.}"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "llama-cli not found at $LLAMA_CLI. Run ./scripts/build_llamacpp.sh first." >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Run ./scripts/download_gguf_models.sh or set MODEL=/path/to/model.gguf." >&2
  exit 1
fi

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -c "$CONTEXT" \
  -n "$PREDICT" \
  -ngl "$N_GPU_LAYERS" \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
  -p "$PROMPT"
