#!/usr/bin/env bash
# Normalize CUDA 12.9 nvcc --version for PyTorch CMake's exact string check.
set -euo pipefail

REAL_NVCC="${REAL_NVCC:-/usr/local/cuda-12.9/bin/nvcc}"

if [[ "$#" -eq 1 && "$1" == "--version" ]]; then
  "${REAL_NVCC}" "$@" | sed 's/release 12\.9, V12\.9\.86/release 12.9, V12.9/'
  exit "${PIPESTATUS[0]}"
fi

exec "${REAL_NVCC}" "$@"
