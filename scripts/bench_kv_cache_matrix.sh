#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/kv_matrix_$STAMP}"

MODELS="${MODELS:-granite:$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf,gemma:$ROOT_DIR/checkpoints/gguf/gemma-4-E2B-it-bf16.gguf}"
LENGTHS="${LENGTHS:-short:512,medium:2048,long:4096}"
KV_MODES="${KV_MODES:-f16,q8_0,q4_0,q2_0,q2_0_hp}"
KV_PAIRS="${KV_PAIRS:-}"
Q2_0_CLIP_RATIO="${Q2_0_CLIP_RATIO:-0.96}"
CONTEXT="${CONTEXT:-8192}"
GEN_TOKENS="${GEN_TOKENS:-${N_GEN:-128}}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
REPETITIONS="${REPETITIONS:-1}"
FLASH_ATTN="${FLASH_ATTN:-1}"
HP_SINK="${HP_SINK:-512}"
HP_RECENT="${HP_RECENT:-2048}"
VRAM_POLL_INTERVAL="${VRAM_POLL_INTERVAL:-0.2}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
MAX_BASELINE_MIB="${MAX_BASELINE_MIB:-1024}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-0}"
GPU_IDLE_TIMEOUT_SEC="${GPU_IDLE_TIMEOUT_SEC:-0}"
GPU_IDLE_POLL_SEC="${GPU_IDLE_POLL_SEC:-5}"

if [[ ! -x "$LLAMA_BENCH" ]]; then
  echo "llama-bench not found at $LLAMA_BENCH. Build first or set LLAMA_BENCH=/path/to/llama-bench." >&2
  exit 1
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  CHECK_BENCH=1 CHECK_PPL=0 CHECK_GPU=1 \
    LLAMA_BENCH="$LLAMA_BENCH" MODELS="$MODELS" "$ROOT_DIR/scripts/check_kv_env.sh"
fi

gpu_process_snapshot() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
}

guard_baseline_mib=""
guard_gpu_processes=""

if [[ "$ALLOW_BUSY_GPU" != "1" ]]; then
  idle_start_sec="$(date +%s)"
  baseline_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 || true)"
  baseline_mib="$(printf '%s\n' "$baseline_raw" | head -1 | tr -d ' ')"
  while :; do
    guard_baseline_mib="$baseline_mib"
    guard_gpu_processes="$(gpu_process_snapshot)"
    if [[ ! "$baseline_mib" =~ ^[0-9]+$ ]]; then
      echo "Could not read current GPU memory baseline: $baseline_raw" >&2
      echo "GPU compute process snapshot:" >&2
      printf '%s\n' "$guard_gpu_processes" >&2
      echo "Set ALLOW_BUSY_GPU=1 to bypass this guard." >&2
      exit 1
    fi
    if (( baseline_mib <= MAX_BASELINE_MIB )); then
      break
    fi
    if [[ "$WAIT_FOR_IDLE_GPU" != "1" ]]; then
      echo "GPU baseline is ${baseline_mib} MiB, above MAX_BASELINE_MIB=${MAX_BASELINE_MIB} MiB." >&2
      echo "Another process or allocator residue may contaminate VRAM results or cause OOM." >&2
      echo "GPU compute process snapshot:" >&2
      printf '%s\n' "$guard_gpu_processes" >&2
      echo "Stop other GPU jobs, set WAIT_FOR_IDLE_GPU=1, raise MAX_BASELINE_MIB, or set ALLOW_BUSY_GPU=1 for smoke runs." >&2
      exit 1
    fi
    now_sec="$(date +%s)"
    if (( GPU_IDLE_TIMEOUT_SEC > 0 && now_sec - idle_start_sec >= GPU_IDLE_TIMEOUT_SEC )); then
      echo "Timed out waiting for idle GPU: baseline is ${baseline_mib} MiB, threshold is ${MAX_BASELINE_MIB} MiB." >&2
      echo "GPU compute process snapshot:" >&2
      printf '%s\n' "$guard_gpu_processes" >&2
      echo "Set ALLOW_BUSY_GPU=1 to bypass this guard for smoke runs." >&2
      exit 1
    fi
    echo "Waiting for idle GPU: baseline ${baseline_mib} MiB > ${MAX_BASELINE_MIB} MiB; sleeping ${GPU_IDLE_POLL_SEC}s." >&2
    echo "GPU compute process snapshot:" >&2
    printf '%s\n' "$guard_gpu_processes" >&2
    sleep "$GPU_IDLE_POLL_SEC"
    baseline_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 || true)"
    baseline_mib="$(printf '%s\n' "$baseline_raw" | head -1 | tr -d ' ')"
  done
fi

if [[ -z "$guard_baseline_mib" ]]; then
  guard_baseline_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 || true)"
  guard_baseline_mib="$(printf '%s\n' "$guard_baseline_raw" | head -1 | tr -d ' ')"
  guard_gpu_processes="$(gpu_process_snapshot)"
fi

mkdir -p "$OUT_DIR"

{
  echo "llama_bench=$LLAMA_BENCH"
  echo "models=$MODELS"
  echo "lengths=$LENGTHS"
  echo "kv_modes=$KV_MODES"
  echo "kv_pairs=${KV_PAIRS:-$KV_MODES}"
  echo "q2_0_clip_ratio=$Q2_0_CLIP_RATIO"
  echo "context=$CONTEXT"
  echo "gen_tokens=$GEN_TOKENS"
  echo "n_gpu_layers=$N_GPU_LAYERS"
  echo "repetitions=$REPETITIONS"
  echo "flash_attn=$FLASH_ATTN"
  echo "hp_sink=$HP_SINK"
  echo "hp_recent=$HP_RECENT"
  echo "vram_poll_interval=$VRAM_POLL_INTERVAL"
  echo "allow_busy_gpu=$ALLOW_BUSY_GPU"
  echo "max_baseline_mib=$MAX_BASELINE_MIB"
  echo "wait_for_idle_gpu=$WAIT_FOR_IDLE_GPU"
  echo "gpu_idle_timeout_sec=$GPU_IDLE_TIMEOUT_SEC"
  echo "gpu_idle_poll_sec=$GPU_IDLE_POLL_SEC"
  echo "guard_baseline_mib=$guard_baseline_mib"
  echo "guard_gpu_processes_begin"
  printf '%s\n' "$guard_gpu_processes"
  echo "guard_gpu_processes_end"
} > "$OUT_DIR/config.txt"

IFS=',' read -ra MODEL_ENTRIES <<< "$MODELS"
IFS=',' read -ra LENGTH_ENTRIES <<< "$LENGTHS"
IFS=',' read -ra KV_ENTRIES <<< "${KV_PAIRS:-$KV_MODES}"

kv_cache_type() {
  local mode="$1"
  if [[ "$mode" == q2_0_* ]]; then
    printf 'q2_0'
  else
    printf '%s' "$mode"
  fi
}

kv_label() {
  local mode_k="$1"
  local mode_v="$2"
  if [[ "$mode_k" == "$mode_v" ]]; then
    printf '%s' "$mode_k"
  else
    printf 'k%s_v%s' "$mode_k" "$mode_v"
  fi
}

kv_needs_hp() {
  local mode_k="$1"
  local mode_v="$2"
  [[ "$mode_k" == "q2_0_hp" || "$mode_v" == "q2_0_hp" ]]
}

kv_needs_owht() {
  local mode_k="$1"
  local mode_v="$2"
  [[ "$mode_k" == q2_0_owht* || "$mode_v" == q2_0_owht* ]]
}

kv_no_hadamard() {
  local mode_k="$1"
  local mode_v="$2"
  [[ "$mode_k" == *nohad* || "$mode_v" == *nohad* ]]
}

kv_needs_clip() {
  local mode_k="$1"
  local mode_v="$2"
  [[ "$mode_k" == *clip* || "$mode_v" == *clip* ]]
}

append_row_metadata() {
  local summary_file="$1"
  {
    echo "kv_mode_k=$kv_mode_k"
    echo "kv_mode_v=$kv_mode_v"
    echo "cache_type_k=$cache_type_k"
    echo "cache_type_v=$cache_type_v"
    echo "llama_kv_hp_sink=$llama_kv_hp_sink"
    echo "llama_kv_hp_recent=$llama_kv_hp_recent"
    echo "llama_kv_q2_0_owht=$llama_kv_q2_0_owht"
    echo "llama_kv_no_hadamard=$llama_kv_no_hadamard"
    echo "llama_kv_clip_ratio=$llama_kv_clip_ratio"
  } >> "$summary_file"
}

for model_entry in "${MODEL_ENTRIES[@]}"; do
  model_name="${model_entry%%:*}"
  model_path="${model_entry#*:}"

  if [[ ! -f "$model_path" ]]; then
    echo "Skipping missing model: $model_name -> $model_path" >&2
    continue
  fi

  for length_entry in "${LENGTH_ENTRIES[@]}"; do
    length_name="${length_entry%%:*}"
    prompt_tokens="${length_entry#*:}"

    for kv_entry in "${KV_ENTRIES[@]}"; do
      if [[ "$kv_entry" == */* ]]; then
        kv_mode_k="${kv_entry%%/*}"
        kv_mode_v="${kv_entry#*/}"
      else
        kv_mode_k="$kv_entry"
        kv_mode_v="$kv_entry"
      fi
      kv_name="$(kv_label "$kv_mode_k" "$kv_mode_v")"
      label="${model_name}_${length_name}_${kv_name}_p${prompt_tokens}_n${GEN_TOKENS}"
      json_file="$OUT_DIR/$label.json"

      echo "Running $label"

      if kv_needs_hp "$kv_mode_k" "$kv_mode_v"; then
        llama_kv_hp_sink="$HP_SINK"
        llama_kv_hp_recent="$HP_RECENT"
      else
        llama_kv_hp_sink=0
        llama_kv_hp_recent=0
      fi
      if kv_needs_owht "$kv_mode_k" "$kv_mode_v"; then
        llama_kv_q2_0_owht=1
        if kv_no_hadamard "$kv_mode_k" "$kv_mode_v"; then
          llama_kv_no_hadamard=1
        else
          llama_kv_no_hadamard=0
        fi
        if kv_needs_clip "$kv_mode_k" "$kv_mode_v"; then
          llama_kv_clip_ratio="$Q2_0_CLIP_RATIO"
        else
          llama_kv_clip_ratio=0
        fi
      else
        llama_kv_q2_0_owht=0
        llama_kv_no_hadamard=0
        llama_kv_clip_ratio=0
      fi
      cache_type_k="$(kv_cache_type "$kv_mode_k")"
      cache_type_v="$(kv_cache_type "$kv_mode_v")"
      env_cmd=(
        env
        LLAMA_KV_HP_SINK="$llama_kv_hp_sink"
        LLAMA_KV_HP_RECENT="$llama_kv_hp_recent"
        LLAMA_KV_Q2_0_OWHT="$llama_kv_q2_0_owht"
        LLAMA_KV_NO_HADAMARD="$llama_kv_no_hadamard"
        LLAMA_KV_CLIP_RATIO="$llama_kv_clip_ratio"
      )

      VRAM_POLL_INTERVAL="$VRAM_POLL_INTERVAL" \
      "$ROOT_DIR/scripts/measure_vram.sh" "$OUT_DIR" "$label" -- \
        "${env_cmd[@]}" "$LLAMA_BENCH" \
          -m "$model_path" \
          -p "$prompt_tokens" \
          -n "$GEN_TOKENS" \
          -ngl "$N_GPU_LAYERS" \
          -fa "$FLASH_ATTN" \
          -r "$REPETITIONS" \
          --cache-type-k "$cache_type_k" \
          --cache-type-v "$cache_type_v" \
          --output json \
          > "$json_file.summary.stdout"

      append_row_metadata "$OUT_DIR/$label.summary.txt"
      mv "$OUT_DIR/$label.stdout.txt" "$json_file"
    done
  done
done

echo
echo "Results written to: $OUT_DIR"
