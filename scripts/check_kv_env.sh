#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
LLAMA_PPL="${LLAMA_PPL:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-perplexity}"
MODELS="${MODELS:-granite:$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf,gemma:$ROOT_DIR/checkpoints/gguf/gemma-4-E2B-it-bf16.gguf}"
CHECK_BENCH="${CHECK_BENCH:-1}"
CHECK_PPL="${CHECK_PPL:-1}"
CHECK_GPU="${CHECK_GPU:-1}"

failures=0

check_ok() {
  printf '[ok] %s\n' "$1"
}

check_fail() {
  printf '[fail] %s\n' "$1"
  failures=$((failures + 1))
}

check_warn() {
  printf '[warn] %s\n' "$1"
}

check_file_exec() {
  local label="$1"
  local path="$2"
  if [[ -x "$path" ]]; then
    check_ok "$label: $path"
  else
    check_fail "$label not executable: $path"
  fi
}

if [[ "$CHECK_BENCH" == "1" ]]; then
  check_file_exec "llama-bench" "$LLAMA_BENCH"
fi

if [[ "$CHECK_PPL" == "1" ]]; then
  check_file_exec "llama-perplexity" "$LLAMA_PPL"
fi

IFS=',' read -ra MODEL_ENTRIES <<< "$MODELS"
for model_entry in "${MODEL_ENTRIES[@]}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"
  if [[ -f "$model_path" ]]; then
    check_ok "model $model_name: $model_path"
  else
    check_fail "missing model $model_name: $model_path"
  fi
done

if [[ "$CHECK_GPU" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    smi_out="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"
    smi_status=$?
    if [[ $smi_status -eq 0 ]]; then
      check_ok "nvidia-smi: $smi_out"
    else
      check_fail "nvidia-smi failed: $smi_out"
    fi
  else
    check_fail "nvidia-smi not found"
  fi
else
  check_warn "skipping GPU/NVML checks because CHECK_GPU=0"
fi

if [[ "$CHECK_BENCH" == "1" && -x "$LLAMA_BENCH" ]]; then
  bench_help="$("$LLAMA_BENCH" --help 2>&1)"
  if grep -q -- '--n-prompt' <<< "$bench_help" && grep -q -- '--n-gen' <<< "$bench_help"; then
    check_ok "llama-bench supports -p/--n-prompt and -n/--n-gen"
  else
    check_fail "llama-bench help does not show expected -p/-n benchmark parameters"
  fi

  if grep -q -- '--ctx-size' <<< "$bench_help"; then
    check_warn "llama-bench exposes --ctx-size; current matrix script still sizes rows with -p/-n"
  else
    check_ok "llama-bench has no --ctx-size; matrix script does not pass -c"
  fi

  if [[ "$CHECK_GPU" == "1" ]]; then
    list_out="$("$LLAMA_BENCH" --list-devices 2>&1)"
    list_status=$?
    if [[ $list_status -eq 0 ]] && ! grep -qi 'failed to initialize CUDA' <<< "$list_out"; then
      check_ok "llama-bench device listing initialized"
    else
      check_fail "llama-bench CUDA/device init failed: $(tr '\n' ' ' <<< "$list_out" | cut -c1-240)"
    fi
  fi
fi

if [[ "$CHECK_PPL" == "1" && -x "$LLAMA_PPL" ]]; then
  ppl_help="$("$LLAMA_PPL" --help 2>&1)"
  if grep -q -- '--cache-type-k' <<< "$ppl_help" && grep -q -- '--chunks' <<< "$ppl_help"; then
    check_ok "llama-perplexity supports cache type and chunk options"
  else
    check_fail "llama-perplexity help does not show expected cache/chunk options"
  fi
fi

if [[ $failures -eq 0 ]]; then
  printf '\npreflight passed\n'
else
  printf '\npreflight failed with %d issue(s)\n' "$failures"
fi

exit "$failures"
