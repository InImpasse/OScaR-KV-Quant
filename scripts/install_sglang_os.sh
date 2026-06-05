#!/usr/bin/env bash
# Editable-install OSCAR's vendored SGLang (sglang-research) for this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGL="${ROOT}/third_party/OSCAR/sglang-research/python"
if [[ ! -f "${SGL}/pyproject.toml" ]]; then
  echo "Missing ${SGL}. Run: git submodule update --init --recursive"
  exit 1
fi
echo "Installing SGLang from ${SGL}"
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "${SGL}"
else
  pip install -e "${SGL}"
fi
echo "Done. Verify: python -c 'import sglang; print(sglang.__file__)'"
