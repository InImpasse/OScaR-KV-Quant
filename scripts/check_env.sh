#!/usr/bin/env bash
# Check host prerequisites for OSCAR-KV-Quant without installing anything.
set -euo pipefail

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
REQUIRED_CUDA_MIN_MINOR=8
REQUIRED_CUDA_MAX_MINOR=9

ok=1

section() {
  echo
  echo "== $1 =="
}

version_ge_py312() {
  local version="$1"
  local major minor
  major="$(echo "$version" | awk -F. '{print $1}')"
  minor="$(echo "$version" | awk -F. '{print $2}')"
  [[ "$major" -gt "$REQUIRED_PYTHON_MAJOR" ]] || {
    [[ "$major" -eq "$REQUIRED_PYTHON_MAJOR" && "$minor" -ge "$REQUIRED_PYTHON_MINOR" ]]
  }
}

section "uv"
if command -v uv >/dev/null 2>&1; then
  uv --version
else
  ok=0
  echo "uv: missing"
  echo "Install suggestion:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  exec \$SHELL"
fi

section "Python 3.12"
if command -v python3.12 >/dev/null 2>&1; then
  python3.12 --version
else
  ok=0
  echo "python3.12: missing"
  echo "Install suggestions:"
  echo "  uv python install 3.12"
  echo "  # or use your distro/package manager to install Python 3.12"
fi

section "Default python3"
if command -v python3 >/dev/null 2>&1; then
  pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
  echo "python3: ${pyver}"
  if ! version_ge_py312 "$pyver"; then
    echo "Note: default python3 is older than 3.12. This is fine if you use scripts/setup_env_uv.sh."
  fi
else
  echo "python3: missing"
fi

section "NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
    ok=0
    echo "nvidia-smi is present but cannot access the GPU."
    echo "Check the NVIDIA Windows driver, WSL GPU passthrough, and whether another policy blocks NVML."
  fi
else
  ok=0
  echo "nvidia-smi: missing"
  echo "Install/update the NVIDIA Windows driver and ensure WSL GPU passthrough is working."
fi

section "CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
  cuda_version="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -1)"
  if [[ -n "$cuda_version" ]]; then
    cuda_major="$(echo "$cuda_version" | awk -F. '{print $1}')"
    cuda_minor="$(echo "$cuda_version" | awk -F. '{print $2}')"
    if [[ "$cuda_major" -ne 12 || "$cuda_minor" -lt "$REQUIRED_CUDA_MIN_MINOR" || "$cuda_minor" -gt "$REQUIRED_CUDA_MAX_MINOR" ]]; then
      ok=0
      echo "CUDA toolkit ${cuda_version} detected, but upstream OSCAR expects CUDA 12.8 or 12.9."
      echo "Upgrade suggestion:"
      echo "  Install CUDA Toolkit 12.8 or 12.9 in WSL."
      echo "  export CUDA_HOME=/usr/local/cuda-12.9"
      echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
      echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}\""
    fi
  fi
else
  ok=0
  echo "nvcc: missing"
  echo "Install CUDA Toolkit 12.8 or 12.9 and ensure nvcc is on PATH."
fi

section "Build tools"
if command -v ninja >/dev/null 2>&1; then
  ninja --version
else
  echo "ninja: missing on PATH"
  echo "The setup script installs the Python ninja wheel into .venv-oscar-kv."
  echo "If SGLang JIT still cannot find ninja, run:"
  echo "  .venv-oscar-kv/bin/python -m pip install ninja"
fi

section "Checkpoints"
for cfg in \
  "checkpoints/granite-4.0-1b-base/config.json" \
  "checkpoints/gemma-4-E2B/config.json"
do
  if [[ -f "$cfg" ]]; then
    echo "found: $cfg"
  else
    echo "missing: $cfg"
    echo "Run ./scripts/download_models.sh if needed. Existing checkpoints are skipped automatically."
  fi
done

section "Summary"
if [[ "$ok" -eq 1 ]]; then
  echo "Host checks passed."
else
  echo "Some host checks failed or need attention. See suggestions above."
  exit 1
fi
