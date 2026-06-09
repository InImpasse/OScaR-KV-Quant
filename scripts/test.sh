#!/usr/bin/env bash
# Run unit tests from the project virtual environment without activation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${ROOT}"
# shellcheck source=lib/repo_paths.sh
source "${ROOT}/scripts/lib/repo_paths.sh"
VENV_DIR="${ROOT}/.venv-oscar-kv"
RUN_SERVER_TESTS=0
RUN_CLI_SMOKE=1
TEST_PORT=31991
TEST_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./scripts/test.sh [options] [unittest discover args]

Options:
  --venv-dir PATH          Virtualenv directory (default: .venv-oscar-kv)
  --unit-only              Run only root tests/ unittest discovery
  --skip-cli-smoke         Skip lightweight wrapper/CLI smoke checks
  --run-server-tests       Run the optional local SGLang dummy server test
  --run-integration        Alias for --run-server-tests
  --test-port PORT         Port for --run-server-tests (default: 31991)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --unit-only) RUN_CLI_SMOKE=0; RUN_SERVER_TESTS=0; shift ;;
    --skip-cli-smoke) RUN_CLI_SMOKE=0; shift ;;
    --run-server-tests) RUN_SERVER_TESTS=1; shift ;;
    --run-integration) RUN_SERVER_TESTS=1; shift ;;
    --test-port) TEST_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) TEST_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi
PY="${VENV_DIR}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}" >&2
  echo "Run ./scripts/setup_env_uv.sh first." >&2
  exit 1
fi

cd "${ROOT}"
setup_runtime_caches
"${PY}" -m unittest discover -s tests "${TEST_ARGS[@]}"

if [[ "${RUN_CLI_SMOKE}" -eq 1 && "${#TEST_ARGS[@]}" -eq 0 ]]; then
  echo "[test] CLI smoke: wrapper help and dry-run paths"
  ./scripts/probe.sh --help >/dev/null
  ./scripts/bench.sh --help >/dev/null
  ./scripts/probe.sh >/dev/null
  ./scripts/bench.sh \
    --profile granite \
    --preset short \
    --modes bf16,int2 \
    --results-dir "${ROOT}/.cache/test-bench-dry-run" \
    --dry-run >/dev/null
fi

if [[ "${RUN_SERVER_TESTS}" -eq 1 ]]; then
  exec "${PY}" tests/test_sglang_server_optional.py \
    --run-server-tests \
    --test-port "${TEST_PORT}"
fi
