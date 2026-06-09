#!/usr/bin/env bash
# Run oscar-kv-probe from the project virtual environment without activation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${ROOT}"
# shellcheck source=lib/repo_paths.sh
source "${ROOT}/scripts/lib/repo_paths.sh"
VENV_DIR="${ROOT}/.venv-oscar-kv"
CUDA_TOOLKIT_HOME=""
PROBE_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/probe.sh [wrapper options] [oscar-kv-probe args]

Wrapper options:
  --venv-dir PATH          Virtualenv directory (default: .venv-oscar-kv)
  --cuda-home PATH         CUDA toolkit directory to prepend to PATH/LD_LIBRARY_PATH
  -h, --help               Show this help and oscar-kv-probe --help if installed
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --cuda-home) CUDA_TOOLKIT_HOME="$2"; shift 2 ;;
    -h|--help) usage; PROBE_ARGS+=("$1"); shift ;;
    *) PROBE_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi
PROBE="${VENV_DIR}/bin/oscar-kv-probe"

if [[ ! -x "${PROBE}" ]]; then
  echo "Missing ${PROBE}" >&2
  echo "Run ./scripts/setup_env_uv.sh first." >&2
  exit 1
fi

if [[ -n "${CUDA_TOOLKIT_HOME}" && ! -x "${CUDA_TOOLKIT_HOME}/bin/nvcc" ]]; then
  echo "Ignoring invalid CUDA toolkit path=${CUDA_TOOLKIT_HOME}; nvcc was not found there." >&2
  CUDA_TOOLKIT_HOME=""
fi
if [[ -z "${CUDA_TOOLKIT_HOME}" ]]; then
  for cuda_candidate in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
    if [[ -x "${cuda_candidate}/bin/nvcc" ]]; then
      CUDA_TOOLKIT_HOME="${cuda_candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_TOOLKIT_HOME}" ]]; then
  export CUDA_HOME="${CUDA_TOOLKIT_HOME}"
  export PATH="${CUDA_TOOLKIT_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
setup_runtime_caches
exec "${PROBE}" "${PROBE_ARGS[@]}"
