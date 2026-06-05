#!/usr/bin/env bash
# Run oscar-kv-probe from the project virtual environment without activation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv-oscar-kv}"
PROBE="${VENV_DIR}/bin/oscar-kv-probe"

if [[ ! -x "${PROBE}" ]]; then
  echo "Missing ${PROBE}" >&2
  echo "Run ./scripts/setup_env_uv.sh first." >&2
  exit 1
fi

if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Ignoring invalid CUDA_HOME=${CUDA_HOME}; nvcc was not found there." >&2
  unset CUDA_HOME
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  for cuda_candidate in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
    if [[ -x "${cuda_candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
exec "${PROBE}" "$@"
