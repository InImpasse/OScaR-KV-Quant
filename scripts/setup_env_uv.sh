#!/usr/bin/env bash
# Create a Python 3.12 uv environment and install OSCAR-KV-Quant + OSCAR's SGLang.
#
# This follows FutureMLS-Lab/OSCAR's environment shape:
#   - one Python env for dump + eval
#   - Python 3.12
#   - CUDA 12.8/12.9 with nvcc on PATH
#   - editable install of third_party/OSCAR/sglang-research/python
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv-oscar-kv"
PYTHON_VERSION="3.12"
INSTALL_SGLANG=1
DOWNLOAD_MODELS=0
RUN_PROBE=1
RESET_ENV=0
CUDA_TOOLKIT_HOME=""

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_env_uv.sh [options]

Options:
  --venv-dir PATH          Target virtualenv directory (default: .venv-oscar-kv)
  --python-version VERSION Python version installed/used by uv (default: 3.12)
  --install-sglang         Install OSCAR's vendored SGLang (default)
  --no-install-sglang      Skip vendored SGLang install
  --download-models        Run scripts/download_models.sh after install
  --no-download-models     Skip model downloads (default)
  --run-probe              Run oscar-kv-probe at the end (default)
  --no-run-probe           Skip final probe
  --cuda-home PATH         CUDA toolkit directory exported before install/probe
  --reset-env              Recreate the virtualenv if it already exists
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --install-sglang) INSTALL_SGLANG=1; shift ;;
    --no-install-sglang) INSTALL_SGLANG=0; shift ;;
    --download-models) DOWNLOAD_MODELS=1; shift ;;
    --no-download-models) DOWNLOAD_MODELS=0; shift ;;
    --run-probe) RUN_PROBE=1; shift ;;
    --no-run-probe) RUN_PROBE=0; shift ;;
    --cuda-home) CUDA_TOOLKIT_HOME="$2"; shift 2 ;;
    --reset-env) RESET_ENV=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, for example:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "Then restart your shell or source ~/.local/bin/env." >&2
  exit 1
fi

if [[ -n "${CUDA_TOOLKIT_HOME}" ]]; then
  export CUDA_HOME="${CUDA_TOOLKIT_HOME}"
  export PATH="${CUDA_TOOLKIT_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "[setup] repo=${ROOT}"
echo "[setup] venv=${VENV_DIR}"
echo "[setup] python=${PYTHON_VERSION}"

cd "${ROOT}"
git submodule update --init --recursive

if [[ -e "${VENV_DIR}" && "${RESET_ENV}" -eq 1 ]]; then
  echo "[setup] removing existing venv because --reset-env was set: ${VENV_DIR}"
  rm -rf "${VENV_DIR}" || {
    echo "[setup] failed to remove ${VENV_DIR}" >&2
    echo "If it is owned by root, run:" >&2
    echo "  sudo rm -rf \"${VENV_DIR}\"" >&2
    echo "Then rerun: ./scripts/setup_env_uv.sh" >&2
    exit 1
  }
fi

if [[ -e "${VENV_DIR}" && ! -w "${VENV_DIR}" ]]; then
  echo "[setup] existing venv is not writable by the current user: ${VENV_DIR}" >&2
  echo "This usually happens if the setup was previously run with sudo." >&2
  echo "Fix one of these ways:" >&2
  echo "  sudo chown -R \"$(id -un):$(id -gn)\" \"${VENV_DIR}\"" >&2
  echo "  # or recreate it:" >&2
  echo "  sudo rm -rf \"${VENV_DIR}\" && ./scripts/setup_env_uv.sh" >&2
  exit 1
fi

echo "[setup] ensuring Python ${PYTHON_VERSION} is available via uv"
uv python install "${PYTHON_VERSION}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "[setup] reusing existing venv: ${VENV_DIR}"
else
  uv venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi
PY="${VENV_DIR}/bin/python"

"${PY}" -V
"${PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${PY}" -m pip install --upgrade pip setuptools wheel
"${PY}" -m pip install -e "${ROOT}"

if [[ "${INSTALL_SGLANG}" == "1" ]]; then
  SGL="${ROOT}/third_party/OSCAR/sglang-research/python"
  if [[ ! -f "${SGL}/pyproject.toml" ]]; then
    echo "[setup] missing ${SGL}; submodule checkout failed" >&2
    exit 1
  fi
  check_sglang_stack() {
    "${PY}" - <<'PYEOF'
import importlib.metadata as md
import importlib.util

required = {
    "torch": "2.9.1",
    "torchaudio": "2.9.1",
    "torchao": "0.9.0",
    "transformers": "5.3.0",
    "kernels": "0.13.0",
    "flashinfer_python": "0.6.7.post3",
    "flashinfer_cubin": "0.6.7.post3",
    "sglang-kernel": "0.4.1",
}

missing_imports = [
    name for name in ("sglang", "flashinfer") if importlib.util.find_spec(name) is None
]
if missing_imports:
    print("[setup] missing imports: " + ", ".join(missing_imports))
    raise SystemExit(1)

mismatches = []
for dist_name, expected in required.items():
    try:
        installed = md.version(dist_name)
    except md.PackageNotFoundError:
        mismatches.append(f"{dist_name}: missing (expected {expected})")
        continue
    # Wheels may include a local version suffix such as +cu129. Upstream pins
    # the public version, so compare that portion.
    public = installed.split("+", 1)[0]
    if public != expected:
        mismatches.append(f"{dist_name}: installed {installed}, expected {expected}")

if mismatches:
    print("[setup] SGLang stack version mismatch:")
    for item in mismatches:
        print(f"  - {item}")
    raise SystemExit(1)

print("[setup] SGLang stack versions match upstream pins")
PYEOF
  }

  if check_sglang_stack
  then
    echo "[setup] skipping SGLang reinstall"
  else
    echo "[setup] installing OSCAR vendored SGLang from ${SGL}"
    "${PY}" -m pip install -e "${SGL}"
    # OSCAR's vendored SGLang leaves `kernels` unpinned. Newer kernels
    # releases require LayerRepository(revision/version), which breaks
    # transformers==5.3.0 during SGLang startup.
    "${PY}" -m pip install "kernels==0.13.0"
    echo "[setup] verifying SGLang stack after install"
    check_sglang_stack
  fi
fi

if [[ "${DOWNLOAD_MODELS}" == "1" ]]; then
  "${ROOT}/scripts/download_models.sh" --venv-dir "${VENV_DIR}"
fi

echo "[setup] verifying installed CLI entry points"
"${VENV_DIR}/bin/oscar-kv-probe" --help >/dev/null
"${VENV_DIR}/bin/oscar-kv-bench" --help >/dev/null

if [[ "${RUN_PROBE}" == "1" ]]; then
  "${VENV_DIR}/bin/oscar-kv-probe"
fi

cat <<EOF

[setup] done

This environment is self-contained at:
  ${VENV_DIR}

Recommended next checks:
  ./scripts/probe.sh --try-dummy-server
  ./scripts/probe.sh --model-path checkpoints/granite-4.0-1b-base --kv-cache-dtype bf16
  ./scripts/bench.sh --profile granite --preset short --modes bf16,int2 --dry-run

Optional interactive activation:
  source "${VENV_DIR}/bin/activate"
  hash -r

If flashinfer / Triton JIT fails, set CUDA_HOME to the CUDA toolkit that matches
your PyTorch CUDA build, e.g.:
  export CUDA_HOME=/usr/local/cuda-12.9
EOF
