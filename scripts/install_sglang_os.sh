#!/usr/bin/env bash
# Editable-install OSCAR's vendored SGLang (sglang-research) for this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGL="${ROOT}/third_party/OSCAR/sglang-research/python"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv-oscar-kv}"
PY="${VENV_DIR}/bin/python"
if [[ ! -f "${SGL}/pyproject.toml" ]]; then
  echo "Missing ${SGL}. Run: git submodule update --init --recursive"
  exit 1
fi
if [[ ! -x "${PY}" ]]; then
  echo "Missing venv at ${VENV_DIR} (expected ${PY})." >&2
  echo "Run: ./scripts/setup_env_uv.sh" >&2
  exit 1
fi
echo "Installing SGLang from ${SGL} into ${VENV_DIR}"
if command -v uv >/dev/null 2>&1; then
  # uv defaults to .venv or an activated venv; this repo uses .venv-oscar-kv.
  uv pip install --python "${PY}" -e "${SGL}"
else
  "${PY}" -m pip install -e "${SGL}"
fi
echo "Done. Verify: \"${PY}\" -c 'import sglang; print(sglang.__file__)'"
