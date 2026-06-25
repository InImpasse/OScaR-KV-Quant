#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSCAR_DIR="$ROOT_DIR/third_party/OSCAR"
BUILD_DIR="${BUILD_DIR:-$OSCAR_DIR/build-cuda}"
LLAMACPP_CMAKE_ARGS="${LLAMACPP_CMAKE_ARGS:--DLLAMA_CURL=OFF -DGGML_CUDA=ON}"

cmake -S "$OSCAR_DIR" -B "$BUILD_DIR" $LLAMACPP_CMAKE_ARGS
cmake --build "$BUILD_DIR" -j "${JOBS:-$(nproc)}" --target llama-cli llama-bench

echo
echo "Built:"
echo "  $BUILD_DIR/bin/llama-cli"
echo "  $BUILD_DIR/bin/llama-bench"
