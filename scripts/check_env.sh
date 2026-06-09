#!/usr/bin/env bash
# Check host prerequisites for OSCAR-KV-Quant without installing anything.
set -euo pipefail

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12

ok=1

# CUDA toolkit on PATH (nvcc): accept CUDA >= 12.8 (e.g. 12.8, 12.9, 12.10+, 13.x).
# PyTorch wheels may be cu128/cu129; if JIT miscompiles, align CUDA_HOME with that build.
cuda_toolkit_accepted() {
  local maj="$1" min="$2"
  [[ "$maj" -eq 12 && "$min" -ge 8 ]] && return 0
  [[ "$maj" -gt 12 ]] && return 0
  return 1
}

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
    if ! cuda_toolkit_accepted "$cuda_major" "$cuda_minor"; then
      ok=0
      echo "CUDA toolkit ${cuda_version} detected; this script requires nvcc for CUDA 12.8 or newer."
      echo "Example (documented upstream stack): install CUDA 12.9 and use:"
      echo "  export CUDA_HOME=/usr/local/cuda-12.9"
      echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
      echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}\""
    elif [[ "$cuda_major" -gt 12 ]] || [[ "$cuda_major" -eq 12 && "$cuda_minor" -gt 9 ]]; then
      echo "Note: nvcc is newer than the upstream-documented 12.8–12.9 line. If FlashInfer/Triton JIT miscompiles, point CUDA_HOME at a toolkit matching your PyTorch CUDA build (often 12.9 for cu129)."
    fi
  fi
else
  ok=0
  echo "nvcc: missing"
  echo "Install CUDA Toolkit 12.8 or newer and ensure nvcc is on PATH."
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
