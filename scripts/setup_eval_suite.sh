#!/usr/bin/env bash
# Install optional dependencies for the upstream OSCAR accuracy suite.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
PY=".venv-oscar-kv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_eval_suite.sh [options]

Options:
  --python PATH   Python executable (default: .venv-oscar-kv/bin/python)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="$(resolve_repo_path "${PY}")"

if [[ ! -x "${PY}" ]]; then
  echo "[setup eval suite] missing python: ${PY}" >&2
  echo "Run ./scripts/setup_env_uv.sh first, or pass --python /path/to/python." >&2
  exit 1
fi

echo "[setup eval suite] installing HumanEval and dataset helpers into ${PY}"
"${PY}" -m pip install "setuptools>=69" "datasets"
if ! "${PY}" -m pip install --no-build-isolation \
  "git+https://github.com/openai/human-eval.git"; then
  echo "[setup eval suite] pip hit HumanEval's legacy console-script metadata."
  echo "[setup eval suite] falling back to a source checkout on PYTHONPATH."
  if [[ ! -d "${REPO_ROOT}/third_party/human-eval/.git" ]]; then
    git clone https://github.com/openai/human-eval.git "${REPO_ROOT}/third_party/human-eval"
  fi
fi

cat <<'EOF'

Optional LiveCodeBench v6 setup:

  git clone https://github.com/LiveCodeBench/LiveCodeBench.git third_party/LiveCodeBench
  cd third_party/LiveCodeBench
  python -m pip install -e .

Then run the Granite eval wrappers with:

  LIVE_CODE_BENCH_ROOT=third_party/LiveCodeBench

The current FutureMLS-Lab/OSCAR submodule documents eval_oscar_lcb.sh, but this
checkout does not include that script, so LiveCodeBench is kept as an external
harness integration.
EOF
