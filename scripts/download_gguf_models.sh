#!/usr/bin/env bash
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-$HOME/models/gguf}"
mkdir -p "$GGUF_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi

download_file() {
  local repo="$1"
  local file="$2"

  echo "Downloading $repo/$file"
  huggingface-cli download "$repo" "$file" --local-dir "$GGUF_DIR"
}

download_file "ggml-org/gemma-4-E2B-it-GGUF" "gemma-4-E2B-it-bf16.gguf"
download_file "ibm-granite/granite-4.0-1b-base-GGUF" "granite-4.0-1b-base-bf16.gguf"

echo
echo "GGUF files are in: $GGUF_DIR"
