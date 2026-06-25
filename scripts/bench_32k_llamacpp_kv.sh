#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR_WAS_SET="${OUT_DIR+x}"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/llamacpp_32k_kv_$STAMP}"

PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
GEN_TOKENS="${GEN_TOKENS:-1}"
REPETITIONS="${REPETITIONS:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-1}"
VRAM_POLL_INTERVAL="${VRAM_POLL_INTERVAL:-0.2}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
MAX_BASELINE_MIB="${MAX_BASELINE_MIB:-1024}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
MAX_PEAK_MIB="${MAX_PEAK_MIB:-0}"
POST_CASE_COOLDOWN_SEC="${POST_CASE_COOLDOWN_SEC:-0}"
POST_CASE_COOLDOWN_POLL_SEC="${POST_CASE_COOLDOWN_POLL_SEC:-2}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-180}"
CASES="${CASES:-baseline_bf16}"
ACK_HEAVY_32K="${ACK_HEAVY_32K:-0}"
ACK_Q2_32K_NOGO="${ACK_Q2_32K_NOGO:-0}"
ACK_Q2_RAMP_GATE_HOLD="${ACK_Q2_RAMP_GATE_HOLD:-0}"
DRY_RUN="${DRY_RUN:-1}"
CUDA_GRAPHS_MODE="${CUDA_GRAPHS_MODE:-auto}"
CUDA_GRAPH_OPT="${CUDA_GRAPH_OPT:-0}"
overall_status=0

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"

case "$CUDA_GRAPHS_MODE" in
  auto|on|off) ;;
  *)
    echo "Unsupported CUDA_GRAPHS_MODE=$CUDA_GRAPHS_MODE. Use auto, on, or off." >&2
    exit 2
    ;;
esac

case "$CUDA_GRAPH_OPT" in
  0|1) ;;
  *)
    echo "Unsupported CUDA_GRAPH_OPT=$CUDA_GRAPH_OPT. Use 0 or 1." >&2
    exit 2
    ;;
esac

if [[ ! "$POST_CASE_COOLDOWN_SEC" =~ ^[0-9]+$ ]]; then
  echo "POST_CASE_COOLDOWN_SEC must be a non-negative integer second value, got: $POST_CASE_COOLDOWN_SEC" >&2
  exit 2
fi
if [[ ! "$POST_CASE_COOLDOWN_POLL_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "POST_CASE_COOLDOWN_POLL_SEC must be a non-negative second value, got: $POST_CASE_COOLDOWN_POLL_SEC" >&2
  exit 2
fi

contains_q2_32k=0
if (( PROMPT_TOKENS >= 32768 )) && [[ "$CASES" == "all" || ",$CASES," == *",plain_int2,"* || ",$CASES," == *",oscar_int2,"* || ",$CASES," == *",turbo2_streamk,"* || ",$CASES," == *",oscar_turbo2_streamk,"* || ",$CASES," == *",turbo2_default,"* ]]; then
  contains_q2_32k=1
fi

if (( contains_q2_32k )); then
  if [[ "$CASES" == "all" || "$CASES" == *,* ]]; then
    echo "Refusing 32k q2/int2 in a multi-case run. Use CASES=plain_int2, CASES=oscar_int2, CASES=turbo2_streamk, CASES=oscar_turbo2_streamk, or CASES=turbo2_default only." >&2
    exit 2
  fi
  if (( GEN_TOKENS != 1 || REPETITIONS != 1 )); then
    echo "Refusing 32k q2/int2 unless GEN_TOKENS=1 and REPETITIONS=1." >&2
    exit 2
  fi
  if [[ "$ACK_HEAVY_32K" != "1" ]]; then
    echo "Refusing heavy 32k q2/int2 run without ACK_HEAVY_32K=1." >&2
    echo "Recommended ramp: PROMPT_TOKENS=8192 first, then 16384, then single 32768 case with CASE_TIMEOUT_SEC set." >&2
    exit 2
  fi
  if [[ "$ACK_Q2_32K_NOGO" != "1" ]]; then
    echo "Refusing 32k q2/int2 after observed NO-GO without ACK_Q2_32K_NOGO=1." >&2
    echo "Observed: oscar_int2 32k timed out at 508.7s with empty JSON. Prefer code/profiler work before re-running." >&2
    exit 2
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    q2_ramp_recommendation="$(python3 "$ROOT_DIR/scripts/report_q2_ramp_gate.py" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["recommendation"])')"
    if [[ "$q2_ramp_recommendation" == "hold_32k_q2" && "$ACK_Q2_RAMP_GATE_HOLD" != "1" ]]; then
      echo "Refusing real 32k q2/int2 while q2 ramp gate is hold_32k_q2." >&2
      echo "Run python3 scripts/report_q2_ramp_gate.py and prefer code/profiler work before repeating the known 32k q2 NO-GO path." >&2
      echo "Set ACK_Q2_RAMP_GATE_HOLD=1 only for a deliberate post-change 32k q2 validation." >&2
      exit 2
    fi
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RUN_PREFLIGHT=0
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "llama-bench not found at $LLAMA_BENCH. Build first or set LLAMA_BENCH=/path/to/llama-bench." >&2
    exit 1
  fi

  if [[ ! -f "$BASE_MODEL" ]]; then
    echo "BASE_MODEL not found: $BASE_MODEL" >&2
    exit 1
  fi

  if [[ ! -f "$OSCAR_MODEL" ]]; then
    echo "OSCAR_MODEL not found: $OSCAR_MODEL" >&2
    exit 1
  fi
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  CHECK_BENCH=1 CHECK_PPL=0 CHECK_GPU=1 MODELS="base:$BASE_MODEL,oscar:$OSCAR_MODEL" \
    LLAMA_BENCH="$LLAMA_BENCH" "$ROOT_DIR/scripts/check_kv_env.sh"
fi

gpu_process_snapshot() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
}

read_gpu_baseline() {
  local raw line mem util
  raw="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1 || true)"
  line="$(printf '%s\n' "$raw" | head -1)"
  mem="$(cut -d, -f1 <<< "$line" | tr -d ' ')"
  util="$(cut -d, -f2 <<< "$line" | tr -d ' ')"
  if [[ ! "$mem" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ ]]; then
    echo "Could not read current GPU memory/utilization baseline: $raw" >&2
    return 1
  fi
  printf '%s %s\n' "$mem" "$util"
}

wait_for_gpu_cooldown() {
  local label="$1"
  local deadline now pair mem util
  if (( POST_CASE_COOLDOWN_SEC <= 0 )) || [[ "$ALLOW_BUSY_GPU" == "1" ]]; then
    return 0
  fi
  deadline=$(( $(date +%s) + POST_CASE_COOLDOWN_SEC ))
  while true; do
    pair="$(read_gpu_baseline)" || return 1
    mem="${pair%% *}"
    util="${pair##* }"
    if (( mem <= MAX_BASELINE_MIB && util <= MAX_GPU_UTIL )); then
      echo "GPU cooldown after $label: ${mem} MiB / ${util}% util"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "GPU did not cool down after $label within ${POST_CASE_COOLDOWN_SEC}s: ${mem} MiB / ${util}% util" >&2
      gpu_process_snapshot >&2
      return 1
    fi
    sleep "$POST_CASE_COOLDOWN_POLL_SEC"
  done
}

if [[ "$DRY_RUN" == "1" ]]; then
  baseline_mib="dry-run"
  baseline_util="dry-run"
else
  baseline_pair="$(read_gpu_baseline)"
  baseline_mib="${baseline_pair%% *}"
  baseline_util="${baseline_pair##* }"

  if [[ "$ALLOW_BUSY_GPU" != "1" ]] && (( baseline_mib > MAX_BASELINE_MIB || baseline_util > MAX_GPU_UTIL )); then
    echo "GPU baseline is ${baseline_mib} MiB / ${baseline_util}% util, above guard MAX_BASELINE_MIB=${MAX_BASELINE_MIB} MiB or MAX_GPU_UTIL=${MAX_GPU_UTIL}%." >&2
    echo "GPU compute process snapshot:" >&2
    gpu_process_snapshot >&2
    echo "Set ALLOW_BUSY_GPU=1 to bypass this guard." >&2
    exit 1
  fi
fi

if [[ "$DRY_RUN" == "1" && -z "$OUT_DIR_WAS_SET" ]]; then
  OUT_DIR="/tmp/llamacpp_32k_kv_dry_run"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT_DIR"
  cat > "$OUT_DIR/config.txt" <<EOF
llama_bench=$LLAMA_BENCH
base_model=$BASE_MODEL
oscar_model=$OSCAR_MODEL
prompt_tokens=$PROMPT_TOKENS
gen_tokens=$GEN_TOKENS
repetitions=$REPETITIONS
n_gpu_layers=$N_GPU_LAYERS
flash_attn=$FLASH_ATTN
vram_poll_interval=$VRAM_POLL_INTERVAL
baseline_mib=$baseline_mib
baseline_util=$baseline_util
max_peak_mib=$MAX_PEAK_MIB
post_case_cooldown_sec=$POST_CASE_COOLDOWN_SEC
post_case_cooldown_poll_sec=$POST_CASE_COOLDOWN_POLL_SEC
plain_int3_mapping=plain_int3 uses the TurboQuant 3-bit KV cache type turbo3/turbo3; Q3_K remains a weight quantization format, not a KV cache type.
turbo2_streamk_gate=LLAMA_TURBO_VEC_STREAM_K=1
cases=$CASES
case_timeout_sec=$CASE_TIMEOUT_SEC
ack_heavy_32k=$ACK_HEAVY_32K
ack_q2_32k_nogo=$ACK_Q2_32K_NOGO
ack_q2_ramp_gate_hold=$ACK_Q2_RAMP_GATE_HOLD
dry_run=$DRY_RUN
cuda_graphs_mode=$CUDA_GRAPHS_MODE
cuda_graph_opt=$CUDA_GRAPH_OPT
EOF
fi

run_case() {
  local label="$1"
  local model="$2"
  local cache_k="$3"
  local cache_v="$4"
  local no_hadamard="$5"
  local clip_ratio="$6"
  local turbo_stream_k="${7:-0}"
  local q2_owht="0"
  if [[ "$cache_k/$cache_v" == "q2_0/q2_0" && "$no_hadamard" == "1" ]]; then
    q2_owht="1"
  fi

  echo "Running $label"
  graph_env=()
  case "$CUDA_GRAPHS_MODE" in
    on)
      graph_env+=(GGML_CUDA_DISABLE_GRAPHS=)
      ;;
    off)
      graph_env+=(GGML_CUDA_DISABLE_GRAPHS=1)
      ;;
  esac
  if [[ "$CUDA_GRAPH_OPT" == "1" ]]; then
    graph_env+=(GGML_CUDA_GRAPH_OPT=1)
  else
    graph_env+=(GGML_CUDA_GRAPH_OPT=0)
  fi

  cmd=(
    env
      VRAM_POLL_INTERVAL="$VRAM_POLL_INTERVAL"
      MAX_PEAK_MIB="$MAX_PEAK_MIB"
    "$ROOT_DIR/scripts/measure_vram.sh" "$OUT_DIR" "$label" --
    env
      "${graph_env[@]}"
      LLAMA_KV_HP_SINK=0
      LLAMA_KV_HP_RECENT=0
      LLAMA_KV_Q2_0_OWHT="$q2_owht"
      LLAMA_KV_NO_HADAMARD="$no_hadamard"
      LLAMA_KV_CLIP_RATIO="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_K="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_V="$([[ "$no_hadamard" == "1" && "$clip_ratio" == "0.96" ]] && printf '0.92' || printf '%s' "$clip_ratio")"
      LLAMA_TURBO_VEC_STREAM_K="$turbo_stream_k"
    "$LLAMA_BENCH"
      -m "$model"
      -p "$PROMPT_TOKENS"
      -n "$GEN_TOKENS"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      -r "$REPETITIONS"
      --cache-type-k "$cache_k"
      --cache-type-v "$cache_v"
      --output json
  )

  if command -v timeout >/dev/null 2>&1 && (( CASE_TIMEOUT_SEC > 0 )); then
    runner=(timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT_SEC}s")
  else
    runner=()
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "${runner[@]}" "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN post-case cooldown: POST_CASE_COOLDOWN_SEC=$POST_CASE_COOLDOWN_SEC POST_CASE_COOLDOWN_POLL_SEC=$POST_CASE_COOLDOWN_POLL_SEC"
    return 0
  fi

  set +e
  "${runner[@]}" "${cmd[@]}" \
      > "$OUT_DIR/$label.measure_stdout.txt"
  status=$?
  set -e
  if (( status != 0 )) && (( overall_status == 0 )); then
    overall_status="$status"
  fi

  if [[ -f "$OUT_DIR/$label.stdout.txt" ]]; then
    mv "$OUT_DIR/$label.stdout.txt" "$OUT_DIR/$label.json"
  fi
  if [[ ! -f "$OUT_DIR/$label.summary.txt" ]]; then
    fallback_peak="$baseline_mib"
    fallback_delta=0
    if [[ ! "$fallback_peak" =~ ^[0-9]+$ ]]; then
      fallback_peak=0
    fi
    {
      echo "label=$label"
      echo "exit_code=$status"
      echo "duration_ms=0"
      echo "baseline_mib=$baseline_mib"
      echo "peak_mib=$fallback_peak"
      echo "delta_mib=$fallback_delta"
      echo "stdout=$OUT_DIR/$label.stdout.txt"
      echo "stderr=$OUT_DIR/$label.stderr.txt"
      echo "metrics=$OUT_DIR/$label.metrics.tsv"
    } > "$OUT_DIR/$label.summary.txt"
  fi
  {
    echo "model=$(basename "$model")"
    echo "cache_type_k=$cache_k"
    echo "cache_type_v=$cache_v"
    echo "llama_kv_no_hadamard=$no_hadamard"
    echo "llama_kv_clip_ratio=$clip_ratio"
    echo "llama_kv_clip_ratio_k=$clip_ratio"
    if [[ "$no_hadamard" == "1" && "$clip_ratio" == "0.96" ]]; then
      echo "llama_kv_clip_ratio_v=0.92"
    else
      echo "llama_kv_clip_ratio_v=$clip_ratio"
    fi
    echo "llama_turbo_vec_stream_k=$turbo_stream_k"
    echo "case_timeout_sec=$CASE_TIMEOUT_SEC"
    echo "cuda_graphs_mode=$CUDA_GRAPHS_MODE"
    echo "cuda_graph_opt=$CUDA_GRAPH_OPT"
  } > "$OUT_DIR/$label.case.txt"

  if ! wait_for_gpu_cooldown "$label"; then
    if (( overall_status == 0 )); then
      overall_status=125
    fi
  fi
}

case_enabled() {
  local needle="$1"
  [[ ",$CASES," == *",$needle,"* || "$CASES" == "all" ]]
}

case_enabled baseline_bf16 && run_case "baseline_bf16_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$BASE_MODEL"  "bf16" "bf16" "0" "0"
case_enabled turbo2_streamk && run_case "turbo2_streamk_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$BASE_MODEL"  "turbo2" "turbo2" "0" "0" "1"
case_enabled oscar_turbo2_streamk && run_case "oscar_turbo2_streamk_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$OSCAR_MODEL" "turbo2" "turbo2" "1" "0.96" "1"
case_enabled turbo2_default && run_case "turbo2_default_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$BASE_MODEL"  "turbo2" "turbo2" "0" "0" "0"
case_enabled turbo3_default && run_case "turbo3_default_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$BASE_MODEL"  "turbo3" "turbo3" "0" "0" "0"
case_enabled oscar_turbo3 && run_case "oscar_turbo3_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$OSCAR_MODEL" "turbo3" "turbo3" "1" "0.96" "0"
case_enabled plain_int3    && run_case "plain_int3_p${PROMPT_TOKENS}_n${GEN_TOKENS}"    "$BASE_MODEL"  "turbo3" "turbo3" "0" "0" "0"
case_enabled plain_int2    && run_case "plain_int2_p${PROMPT_TOKENS}_n${GEN_TOKENS}"    "$BASE_MODEL"  "q2_0" "q2_0" "0" "0"
case_enabled oscar_int2    && run_case "oscar_int2_p${PROMPT_TOKENS}_n${GEN_TOKENS}"    "$OSCAR_MODEL" "q2_0" "q2_0" "1" "0"
case_enabled oscar_kq4_vq2 && run_case "oscar_kq4_vq2_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$OSCAR_MODEL" "q4_0" "q2_0" "1" "0"
case_enabled oscar_kq4_vturbo3 && run_case "oscar_kq4_vturbo3_p${PROMPT_TOKENS}_n${GEN_TOKENS}" "$OSCAR_MODEL" "q4_0" "turbo3" "1" "0"
case_enabled oscar_int4    && run_case "oscar_int4_p${PROMPT_TOKENS}_n${GEN_TOKENS}"    "$OSCAR_MODEL" "q4_0" "q4_0" "1" "0.96"
case_enabled plain_int4    && run_case "plain_int4_p${PROMPT_TOKENS}_n${GEN_TOKENS}"    "$BASE_MODEL"  "q4_0" "q4_0" "0" "0"

if [[ "$DRY_RUN" != "1" ]]; then
  if ! python3 "$ROOT_DIR/scripts/summarize_32k_llamacpp_kv.py" "$OUT_DIR"; then
    echo "warning: failed to summarize $OUT_DIR" >&2
  fi
fi

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no results written."
else
  echo "Results written to: $OUT_DIR"
fi
exit "$overall_status"
