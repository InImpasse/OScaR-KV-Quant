#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_PPL="${LLAMA_PPL:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-perplexity}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/kv_ppl_$STAMP}"

MODELS="${MODELS:-granite:$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf,gemma:$ROOT_DIR/checkpoints/gguf/gemma-4-E2B-it-bf16.gguf}"
CONTEXTS="${CONTEXTS:-short:512,medium:2048,long:4096}"
KV_MODES="${KV_MODES:-f16,q8_0,q4_0,q2_0,q2_0_hp}"
KV_PAIRS="${KV_PAIRS:-}"
Q2_0_CLIP_RATIO="${Q2_0_CLIP_RATIO:-0.96}"
CORPUS="${CORPUS:-}"
CHUNKS="${CHUNKS:-8}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HP_SINK="${HP_SINK:-512}"
HP_RECENT="${HP_RECENT:-2048}"
VRAM_POLL_INTERVAL="${VRAM_POLL_INTERVAL:-0.2}"
MEASURE_VRAM="${MEASURE_VRAM:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
DRY_RUN="${DRY_RUN:-1}"
ACK_PPL_MATRIX="${ACK_PPL_MATRIX:-0}"
MAX_BASELINE_MIB="${MAX_BASELINE_MIB:-1024}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-0}"
GPU_IDLE_TIMEOUT_SEC="${GPU_IDLE_TIMEOUT_SEC:-0}"
GPU_IDLE_POLL_SEC="${GPU_IDLE_POLL_SEC:-5}"
PPL_OUTPUT_RE='(Final estimate:[[:space:]]*)?PPL[[:space:]]*='

IFS=',' read -ra MODEL_ENTRIES <<< "$MODELS"
IFS=',' read -ra CONTEXT_ENTRIES <<< "$CONTEXTS"
IFS=',' read -ra KV_ENTRIES <<< "${KV_PAIRS:-$KV_MODES}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN config:"
  echo "  llama_perplexity=$LLAMA_PPL"
  echo "  models=$MODELS"
  echo "  contexts=$CONTEXTS"
  echo "  kv_modes=$KV_MODES"
  echo "  kv_pairs=${KV_PAIRS:-$KV_MODES}"
  echo "  corpus=${CORPUS:-<unset>}"
  echo "  chunks=$CHUNKS"
  echo "  batch_size=$BATCH_SIZE"
  echo "  n_gpu_layers=$N_GPU_LAYERS"
  echo "  flash_attn=$FLASH_ATTN"
  echo "  measure_vram=$MEASURE_VRAM"
  echo "  output_dir=$OUT_DIR"
  echo
  for model_entry in "${MODEL_ENTRIES[@]}"; do
    model_name="${model_entry%%:*}"
    model_path="${model_entry#*:}"
    for context_entry in "${CONTEXT_ENTRIES[@]}"; do
      context_name="${context_entry%%:*}"
      context_size="${context_entry#*:}"
      for kv_entry in "${KV_ENTRIES[@]}"; do
        if [[ "$kv_entry" == */* ]]; then
          kv_mode_k="${kv_entry%%/*}"
          kv_mode_v="${kv_entry#*/}"
        else
          kv_mode_k="$kv_entry"
          kv_mode_v="$kv_entry"
        fi
        if [[ "$kv_mode_k" == q2_0_* ]]; then
          cache_type_k="q2_0"
        else
          cache_type_k="$kv_mode_k"
        fi
        if [[ "$kv_mode_v" == q2_0_* ]]; then
          cache_type_v="q2_0"
        else
          cache_type_v="$kv_mode_v"
        fi
        if [[ "$kv_mode_k" == "q2_0_hp" || "$kv_mode_v" == "q2_0_hp" ]]; then
          dry_hp_sink="$HP_SINK"
          dry_hp_recent="$HP_RECENT"
        else
          dry_hp_sink=0
          dry_hp_recent=0
        fi
        if [[ "$kv_mode_k" == q2_0_owht* || "$kv_mode_v" == q2_0_owht* ]]; then
          dry_owht=1
        else
          dry_owht=0
        fi
        if [[ "$kv_mode_k" == *nohad* || "$kv_mode_v" == *nohad* ]]; then
          dry_no_hadamard=1
        else
          dry_no_hadamard=0
        fi
        if [[ "$kv_mode_k" == *clip* || "$kv_mode_v" == *clip* ]]; then
          dry_clip_ratio="$Q2_0_CLIP_RATIO"
        else
          dry_clip_ratio=0
        fi
        if [[ "$kv_mode_k" == "$kv_mode_v" ]]; then
          kv_name="$kv_mode_k"
        else
          kv_name="k${kv_mode_k}_v${kv_mode_v}"
        fi
        label="${model_name}_${context_name}_${kv_name}_c${context_size}_chunks${CHUNKS}"
        cmd=(
          env
          LLAMA_KV_HP_SINK="$dry_hp_sink"
          LLAMA_KV_HP_RECENT="$dry_hp_recent"
          LLAMA_KV_Q2_0_OWHT="$dry_owht"
          LLAMA_KV_NO_HADAMARD="$dry_no_hadamard"
          LLAMA_KV_CLIP_RATIO="$dry_clip_ratio"
          "$LLAMA_PPL"
          -m "$model_path"
          -f "${CORPUS:-<unset>}"
          -c "$context_size"
          -b "$BATCH_SIZE"
          -ngl "$N_GPU_LAYERS"
          -fa "$FLASH_ATTN"
          --chunks "$CHUNKS"
          --cache-type-k "$cache_type_k"
          --cache-type-v "$cache_type_v"
        )
        printf 'DRY_RUN command:'
        if [[ "$MEASURE_VRAM" == "1" ]]; then
          printf ' %q' "$ROOT_DIR/scripts/measure_vram.sh" "$OUT_DIR" "$label" -- "${cmd[@]}"
        else
          printf ' %q' "${cmd[@]}"
        fi
        printf '\n'
      done
    done
  done
  echo
  echo "Dry run complete; no executable checks, corpus checks, preflight, GPU checks, or results written. Set DRY_RUN=0 ACK_PPL_MATRIX=1 to run intentionally."
  exit 0
fi

if [[ "$ACK_PPL_MATRIX" != "1" ]]; then
  echo "Refusing PPL matrix without ACK_PPL_MATRIX=1." >&2
  echo "Use DRY_RUN=1 to inspect commands or ACK_PPL_MATRIX=1 DRY_RUN=0 to run intentionally." >&2
  exit 1
fi

if [[ ! -x "$LLAMA_PPL" ]]; then
  echo "llama-perplexity not found at $LLAMA_PPL." >&2
  echo "Build it first, for example: cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-perplexity" >&2
  exit 1
fi

if [[ -z "$CORPUS" || ! -f "$CORPUS" ]]; then
  echo "Set CORPUS=/path/to/text corpus for perplexity evaluation." >&2
  exit 1
fi

corpus_bytes="$(wc -c < "$CORPUS" | tr -d ' ')"
corpus_sha256="$(sha256sum "$CORPUS" | awk '{print $1}')"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  if [[ "$MEASURE_VRAM" == "1" ]]; then
    preflight_gpu=1
  else
    preflight_gpu=0
  fi
  CHECK_BENCH=0 CHECK_PPL=1 CHECK_GPU="$preflight_gpu" \
    LLAMA_PPL="$LLAMA_PPL" MODELS="$MODELS" "$ROOT_DIR/scripts/check_kv_env.sh"
fi

gpu_process_snapshot() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
}

guard_baseline_mib=""
guard_gpu_processes=""

if [[ "$MEASURE_VRAM" == "1" && "$ALLOW_BUSY_GPU" != "1" ]]; then
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
      echo "Set MEASURE_VRAM=0 for CPU/no-VRAM PPL runs, or ALLOW_BUSY_GPU=1 to bypass this guard." >&2
      exit 1
    fi
    if (( baseline_mib <= MAX_BASELINE_MIB )); then
      break
    fi
    if [[ "$WAIT_FOR_IDLE_GPU" != "1" ]]; then
      echo "GPU baseline is ${baseline_mib} MiB, above MAX_BASELINE_MIB=${MAX_BASELINE_MIB} MiB." >&2
      echo "Another process or allocator residue may contaminate PPL VRAM results or cause OOM." >&2
      echo "GPU compute process snapshot:" >&2
      printf '%s\n' "$guard_gpu_processes" >&2
      echo "Stop other GPU jobs, set WAIT_FOR_IDLE_GPU=1, raise MAX_BASELINE_MIB, set MEASURE_VRAM=0, or set ALLOW_BUSY_GPU=1 for smoke runs." >&2
      exit 1
    fi
    now_sec="$(date +%s)"
    if (( GPU_IDLE_TIMEOUT_SEC > 0 && now_sec - idle_start_sec >= GPU_IDLE_TIMEOUT_SEC )); then
      echo "Timed out waiting for idle GPU: baseline is ${baseline_mib} MiB, threshold is ${MAX_BASELINE_MIB} MiB." >&2
      echo "GPU compute process snapshot:" >&2
      printf '%s\n' "$guard_gpu_processes" >&2
      echo "Set MEASURE_VRAM=0 for no-VRAM PPL runs, or ALLOW_BUSY_GPU=1 to bypass this guard for smoke runs." >&2
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

if [[ "$MEASURE_VRAM" == "1" && -z "$guard_baseline_mib" ]]; then
  guard_baseline_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 || true)"
  guard_baseline_mib="$(printf '%s\n' "$guard_baseline_raw" | head -1 | tr -d ' ')"
  guard_gpu_processes="$(gpu_process_snapshot)"
fi

mkdir -p "$OUT_DIR"

{
  echo "llama_perplexity=$LLAMA_PPL"
  echo "models=$MODELS"
  echo "contexts=$CONTEXTS"
  echo "kv_modes=$KV_MODES"
  echo "kv_pairs=${KV_PAIRS:-$KV_MODES}"
  echo "q2_0_clip_ratio=$Q2_0_CLIP_RATIO"
  echo "corpus=$CORPUS"
  echo "corpus_bytes=$corpus_bytes"
  echo "corpus_sha256=$corpus_sha256"
  echo "chunks=$CHUNKS"
  echo "n_gpu_layers=$N_GPU_LAYERS"
  echo "flash_attn=$FLASH_ATTN"
  echo "batch_size=$BATCH_SIZE"
  echo "hp_sink=$HP_SINK"
  echo "hp_recent=$HP_RECENT"
  echo "measure_vram=$MEASURE_VRAM"
  echo "vram_poll_interval=$VRAM_POLL_INTERVAL"
  echo "dry_run=$DRY_RUN"
  echo "ack_ppl_matrix=$ACK_PPL_MATRIX"
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

  for context_entry in "${CONTEXT_ENTRIES[@]}"; do
    context_name="${context_entry%%:*}"
    context_size="${context_entry#*:}"

    for kv_entry in "${KV_ENTRIES[@]}"; do
      if [[ "$kv_entry" == */* ]]; then
        kv_mode_k="${kv_entry%%/*}"
        kv_mode_v="${kv_entry#*/}"
      else
        kv_mode_k="$kv_entry"
        kv_mode_v="$kv_entry"
      fi
      kv_name="$(kv_label "$kv_mode_k" "$kv_mode_v")"
      label="${model_name}_${context_name}_${kv_name}_c${context_size}_chunks${CHUNKS}"
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

      cmd=(
        "${env_cmd[@]}" "$LLAMA_PPL"
        -m "$model_path"
        -f "$CORPUS"
        -c "$context_size"
        -b "$BATCH_SIZE"
        -ngl "$N_GPU_LAYERS"
        -fa "$FLASH_ATTN"
        --chunks "$CHUNKS"
        --cache-type-k "$cache_type_k"
        --cache-type-v "$cache_type_v"
      )

      if [[ "$MEASURE_VRAM" == "1" ]]; then
        REQUIRE_PPL_OUTPUT=1 \
        VRAM_POLL_INTERVAL="$VRAM_POLL_INTERVAL" \
        "$ROOT_DIR/scripts/measure_vram.sh" "$OUT_DIR" "$label" -- "${cmd[@]}"
        append_row_metadata "$OUT_DIR/$label.summary.txt"
      else
        start_ns="$(date +%s%N)"
        set +e
        "${cmd[@]}" > "$OUT_DIR/$label.stdout.txt" 2> "$OUT_DIR/$label.stderr.txt"
        exit_code=$?
        set -e
        end_ns="$(date +%s%N)"
        if [[ "$exit_code" -eq 0 ]] && ! grep -Eq "$PPL_OUTPUT_RE" "$OUT_DIR/$label.stdout.txt" "$OUT_DIR/$label.stderr.txt"; then
          exit_code=2
        fi
        {
          echo "label=$label"
          echo "exit_code=$exit_code"
          echo "duration_ms=$(( (end_ns - start_ns) / 1000000 ))"
          echo "stdout=$OUT_DIR/$label.stdout.txt"
          echo "stderr=$OUT_DIR/$label.stderr.txt"
        } > "$OUT_DIR/$label.summary.txt"
        append_row_metadata "$OUT_DIR/$label.summary.txt"
        if [[ "$exit_code" -ne 0 ]]; then
          exit "$exit_code"
        fi
      fi
    done
  done
done

echo
echo "Results written to: $OUT_DIR"
