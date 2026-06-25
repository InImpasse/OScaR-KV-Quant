#!/usr/bin/env bash
# Profile q2/q2 flash-attn path. Tries Nsight first; falls back to segment microbench.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
MODEL="${MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
NCU_BIN="${NCU_BIN:-${CUDA_HOME}/bin/ncu}"
NSYS_BIN="${NSYS_BIN:-${CUDA_HOME}/bin/nsys}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_DIR:-$ROOT_DIR/runs/q2_profile_${STAMP}}"
DRY_RUN="${DRY_RUN:-1}"
Q2_PROFILE_GPU_SNAPSHOT="${Q2_PROFILE_GPU_SNAPSHOT:-0}"
mkdir -p "$OUT"

# WSL: driver shim + toolkit libs (see AGENTS.md)
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

{
  echo "timestamp=$STAMP"
  echo "model=$MODEL"
  echo "bench=$BENCH"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  if [[ "$Q2_PROFILE_GPU_SNAPSHOT" == "1" ]]; then
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || true
  else
    echo "gpu_snapshot=skipped; set Q2_PROFILE_GPU_SNAPSHOT=1 for read-only nvidia-smi"
  fi
  command -v "$NCU_BIN" && "$NCU_BIN" --version | head -1 || true
  command -v "$NSYS_BIN" && "$NSYS_BIN" --version | head -1 || true
} > "$OUT/env.txt"

BENCH_CMD=("$BENCH" -m "$MODEL" -p 512 -n 1 -r 1 -ngl 999 -fa 1 --cache-type-k q2_0 --cache-type-v q2_0)

if [[ "$DRY_RUN" == "1" ]]; then
  {
    echo "dry_run=1"
    printf 'bench_cmd='
    printf ' %q' "${BENCH_CMD[@]}"
    printf '\n'
    echo "ncu_bin=$NCU_BIN"
    echo "nsys_bin=$NSYS_BIN"
    echo "out=$OUT"
    echo "Set DRY_RUN=0 to attempt profiler execution."
  } | tee "$OUT/dry_run.txt"
  echo "OUT=$OUT"
  exit 0
fi

echo "=== preflight ===" | tee "$OUT/preflight_summary.txt"
OUT_DIR="$OUT" "$ROOT_DIR/scripts/ncu_wsl_preflight.sh" | tee -a "$OUT/preflight_summary.txt" || true
preflight_rc=${PIPESTATUS[0]:-0}
echo "preflight_rc=$preflight_rc" >> "$OUT/preflight_summary.txt"

echo "=== ncu attempt ===" | tee "$OUT/ncu.log"
if [[ -x "$NCU_BIN" && "$preflight_rc" -eq 0 ]]; then
  if timeout 180 "$NCU_BIN" --target-processes all \
      --kernel-name-base demangled \
      --kernel-name regex:flash_attn \
      --launch-count 3 \
      --section SpeedOfLight \
      --csv --log-file "$OUT/ncu.csv" \
      "${BENCH_CMD[@]}" > "$OUT/ncu_stdout.txt" 2>>"$OUT/ncu.log"; then
    echo "ncu_status=ok" >> "$OUT/ncu.log"
  else
    echo "ncu_status=failed exit=$?" >> "$OUT/ncu.log"
    tail -20 "$OUT/ncu.log" || true
  fi
else
  if [[ "$preflight_rc" -ne 0 ]]; then
    echo "ncu_status=skipped preflight_rc=$preflight_rc (see preflight_summary.txt; fix Windows GPU perf counters)" >> "$OUT/ncu.log"
  else
    echo "ncu_status=missing" >> "$OUT/ncu.log"
  fi
fi

echo "=== nsys attempt ===" | tee "$OUT/nsys.log"
if [[ -x "$NSYS_BIN" && "$preflight_rc" -eq 0 ]]; then
  if timeout 180 "$NSYS_BIN" profile --force-overwrite=true \
      --trace=cuda,nvtx --cuda-graph-trace=node \
      --cuda-event-trace=false \
      --sample=none --cpuctxsw=none \
      -o "$OUT/nsys_q2" \
      "${BENCH_CMD[@]}" > "$OUT/nsys_stdout.txt" 2>>"$OUT/nsys.log"; then
    echo "nsys_status=ok" >> "$OUT/nsys.log"
    if nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
        "$OUT/nsys_q2.nsys-rep" > "$OUT/nsys_stats.txt" 2>>"$OUT/nsys.log"; then
      if rg -q "does not contain CUDA kernel data" "$OUT/nsys_stats.txt"; then
        echo "nsys_kernel_data=missing" >> "$OUT/nsys.log"
      else
        echo "nsys_kernel_data=ok" >> "$OUT/nsys.log"
        rg -n "flash_attn|ggml|Total|Time" "$OUT/nsys_stats.txt" | head -40 >> "$OUT/nsys.log" || true
      fi
    fi
  else
    echo "nsys_status=failed exit=$?" >> "$OUT/nsys.log"
    tail -20 "$OUT/nsys.log" || true
  fi
else
  if [[ "$preflight_rc" -ne 0 ]]; then
    echo "nsys_status=skipped preflight_rc=$preflight_rc" >> "$OUT/nsys.log"
  else
    echo "nsys_status=missing" >> "$OUT/nsys.log"
  fi
fi

echo "=== segment microbench fallback ===" | tee "$OUT/fallback.log"
OUT_DIR="$OUT/segments" "$ROOT_DIR/scripts/q2_segment_bench.sh" | tee -a "$OUT/fallback.log"

echo "OUT=$OUT"
