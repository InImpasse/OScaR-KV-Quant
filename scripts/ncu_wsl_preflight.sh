#!/usr/bin/env bash
# Diagnose Nsight Compute / Systems on WSL2 and print actionable fixes.
# Exit: 0 = profilers OK, 1 = needs Windows-side fix, 2 = hard failure / missing tools.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT_DIR:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
WSL_LIB="/usr/lib/wsl/lib"
NCU_BIN="${NCU_BIN:-}"
NSYS_BIN="${NSYS_BIN:-}"

if [[ -z "$NCU_BIN" ]]; then
  for c in "$CUDA_HOME/bin/ncu" /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
    [[ -x "$c" ]] && NCU_BIN="$c" && break
  done
fi
if [[ -z "$NSYS_BIN" ]]; then
  for c in "$CUDA_HOME/bin/nsys" /usr/local/cuda/bin/nsys; do
    [[ -x "$c" ]] && NSYS_BIN="$c" && break
  done
fi

export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${WSL_LIB}:${LD_LIBRARY_PATH:-}"

log() { echo "$@"; [[ -n "$OUT" ]] && echo "$@" >> "$OUT/preflight.log"; }
mkdir -p "${OUT:-/dev/null}" 2>/dev/null || true
: > "${OUT:+$OUT/preflight.log}"

need_windows_fix=0
hard_fail=0

log "=== ncu WSL preflight ==="
log "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "CUDA_HOME=$CUDA_HOME"
log "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

if [[ ! -d "$WSL_LIB" ]]; then
  log "ERROR: $WSL_LIB missing — not running inside WSL2?"
  hard_fail=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | while read -r line; do
    log "gpu: $line"
  done
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
  cuda_max="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  log "driver_version=$drv cuda_max=$cuda_max"
else
  log "ERROR: nvidia-smi not found"
  hard_fail=1
fi

if [[ -x "$WSL_LIB/nvidia-smi" ]]; then
  wsl_smi_out="$("$WSL_LIB/nvidia-smi" --version 2>/dev/null || true)"
  wsl_nvml_ver="$(printf '%s\n' "$wsl_smi_out" | sed -n 's/^NVIDIA-SMI version[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)"
  wsl_drv_ver="$(printf '%s\n' "$wsl_smi_out" | sed -n 's/^DRIVER version[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)"
  log "wsl_stub_nvml=$wsl_nvml_ver wsl_stub_driver=$wsl_drv_ver"
  if [[ -n "${drv:-}" && -n "$wsl_drv_ver" && "$drv" != "$wsl_drv_ver" ]]; then
    log "WARN: nvidia-smi driver ($drv) != WSL stub DRIVER version ($wsl_drv_ver)"
    log "      Run 'wsl --shutdown' in Windows PowerShell, then reopen WSL."
    need_windows_fix=1
  fi
fi

tk_ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"
log "toolkit_nvcc=$tk_ver"
if [[ -n "${cuda_max:-}" && -n "$tk_ver" ]]; then
  cuda_major="${cuda_max%%.*}"
  tk_major="${tk_ver%%.*}"
  if [[ "$cuda_major" != "$tk_major" ]]; then
    log "WARN: Driver reports CUDA $cuda_max but toolkit is $tk_ver."
    log "      Consider: sudo apt install cuda-nsight-compute-13-1  (or nsight-compute-2025.4.1)"
    need_windows_fix=1
  fi
fi

# Minimal CUDA smoke (compile once, reuse)
SMOKE="/tmp/oscar_ncu_smoke"
if [[ ! -x "$SMOKE" ]] && command -v nvcc >/dev/null 2>&1; then
  cat > /tmp/oscar_ncu_smoke.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void k(int *a) { if (threadIdx.x == 0) *a = 1; }
int main() {
    int *d; cudaMalloc(&d, sizeof(int));
    k<<<1, 1>>>(d); cudaDeviceSynchronize();
    cudaFree(d); return 0;
}
EOF
  nvcc -Wno-deprecated-gpu-targets -o "$SMOKE" /tmp/oscar_ncu_smoke.cu 2>/dev/null || true
fi

# CUPTI sanity (API init only)
if [[ -x "$SMOKE" ]]; then
  if LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$SMOKE" >/dev/null 2>&1; then
    log "cuda_smoke=ok"
  else
    log "cuda_smoke=fail"
    hard_fail=1
  fi
fi

# ncu probe
if [[ -x "$NCU_BIN" && -x "$SMOKE" ]]; then
  log "ncu_bin=$NCU_BIN"
  ncu_out="$(mktemp)"
  if timeout 30 "$NCU_BIN" --launch-count 1 "$SMOKE" >"$ncu_out" 2>&1; then
    log "ncu_smoke=ok"
  else
    log "ncu_smoke=fail"
    if rg -q "ERR_NVGPUCTRPERM" "$ncu_out"; then
      log "ncu_error=ERR_NVGPUCTRPERM (GPU perf counter permission)"
      need_windows_fix=1
    elif rg -q "LibraryNotLoaded" "$ncu_out"; then
      log "ncu_error=LibraryNotLoaded (driver profiler API unavailable)"
      need_windows_fix=1
    else
      log "ncu_error=unknown"
      tail -5 "$ncu_out" | while read -r l; do log "  $l"; done
      need_windows_fix=1
    fi
  fi
  rm -f "$ncu_out"
elif [[ ! -x "$NCU_BIN" ]]; then
  log "ncu_bin=missing"
  hard_fail=1
fi

# nsys probe
if [[ -x "$NSYS_BIN" && -x "$SMOKE" ]]; then
  log "nsys_bin=$NSYS_BIN"
  nsys_rep="$(mktemp -u)"
  if timeout 45 "$NSYS_BIN" profile --force-overwrite=true --trace=cuda --sample=none \
      --cuda-event-trace=false -o "$nsys_rep" "$SMOKE" >/dev/null 2>&1; then
    if nsys stats --report cuda_gpu_kern_sum "${nsys_rep}.nsys-rep" 2>&1 | rg -q "does not contain CUDA kernel data"; then
      log "nsys_smoke=fail (no CUDA kernel data in report)"
      need_windows_fix=1
    else
      log "nsys_smoke=ok"
    fi
  else
    log "nsys_smoke=fail (profile command error)"
    need_windows_fix=1
  fi
  rm -f "${nsys_rep}.nsys-rep" "${nsys_rep}.sqlite" 2>/dev/null || true
elif [[ ! -x "$NSYS_BIN" ]]; then
  log "nsys_bin=missing"
fi

if [[ "$need_windows_fix" -eq 1 ]]; then
  log ""
  log "=== Windows-side fix (required for ncu/nsys on WSL) ==="
  log "1. NVIDIA Control Panel → Desktop → Enable Developer Settings"
  log "2. Developer → Manage GPU Performance Counters"
  log "   → 'Allow access to the GPU performance counter to all users'"
  log "3. PowerShell (Admin): wsl --shutdown"
  log "4. Reopen WSL and rerun: scripts/ncu_wsl_preflight.sh"
  log ""
  log "Optional version alignment (if driver reports CUDA 13.x):"
  log "  sudo apt install cuda-nsight-compute-13-1   # or nsight-compute-2025.4.1"
  log "  export NCU_BIN=/usr/local/cuda-13.1/bin/ncu"
  log ""
  log "Reference: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters"
fi

if [[ "$hard_fail" -eq 1 ]]; then
  exit 2
fi
if [[ "$need_windows_fix" -eq 1 ]]; then
  exit 1
fi
exit 0
