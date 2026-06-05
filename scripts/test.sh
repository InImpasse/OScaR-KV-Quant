#!/usr/bin/env bash
# Run unit tests from the project virtual environment without activation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv-oscar-kv}"
PY="${VENV_DIR}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}" >&2
  echo "Run ./scripts/setup_env_uv.sh first." >&2
  exit 1
fi

cd "${ROOT}"
exec "${PY}" -m unittest discover -s tests "$@"
