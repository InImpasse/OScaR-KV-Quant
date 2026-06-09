#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv-oscar-kv"
SWEEP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    *) SWEEP_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi
SWEEP="${VENV_DIR}/bin/oscar-kv-config-sweep"
if [[ ! -x "${SWEEP}" ]]; then
  echo "Missing ${SWEEP}; run ./scripts/setup_env_uv.sh first." >&2
  exit 1
fi
exec "${SWEEP}" "${SWEEP_ARGS[@]}"
