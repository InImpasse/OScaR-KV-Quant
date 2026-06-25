#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/mixed_vec_int2_ramp_$STAMP}"
DRY_RUN="${DRY_RUN:-1}"
ACK_MIXED_VEC_RAMP="${ACK_MIXED_VEC_RAMP:-0}"
VARIANT="${VARIANT:-oscar_int2}"
MODE="${MODE:-all}"
INCLUDE_6144="${INCLUDE_6144:-0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
GEN_TOKENS="${GEN_TOKENS:-1}"
REPETITIONS="${REPETITIONS:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-1}"
VRAM_POLL_INTERVAL="${VRAM_POLL_INTERVAL:-0.5}"
MAX_PEAK_MIB="${MAX_PEAK_MIB:-7000}"
POST_CASE_COOLDOWN_SEC="${POST_CASE_COOLDOWN_SEC:-30}"
POST_CASE_COOLDOWN_POLL_SEC="${POST_CASE_COOLDOWN_POLL_SEC:-2}"

declare -A TIMEOUT_BY_PROMPT=(
  [512]=45
  [2048]=60
  [4096]=75
  [6144]=82
  [8192]=90
)

timeout_for() {
  local mode="$1"
  local prompt="$2"
  local base="${TIMEOUT_BY_PROMPT[$prompt]:-120}"
  if [[ "$mode" != "pure" ]]; then
    base=$(( base * 3 ))
  fi
  echo "$base"
}

if [[ "$DRY_RUN" != "1" && "$ACK_MIXED_VEC_RAMP" != "1" ]]; then
  echo "Refusing mixed vec INT2 ramp without ACK_MIXED_VEC_RAMP=1." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_BENCH" ]] || { echo "llama-bench not found: $LLAMA_BENCH" >&2; exit 1; }
  mkdir -p "$OUT_DIR"
fi

prompts=()
if (( PROMPT_TOKENS > 0 )); then
  prompts=("$PROMPT_TOKENS")
else
  prompts=(512 2048 4096)
  if [[ "$INCLUDE_6144" == "1" ]]; then
    prompts+=(6144)
  fi
  prompts+=(8192)
fi

modes=()
case "$MODE" in
  all) modes=(pure mixed_fused mixed_vec) ;;
  pure|mixed_fused|mixed_vec) modes=("$MODE") ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac

model="$OSCAR_MODEL"
cache_k="q2_0"
cache_v="q2_0"
no_hadamard="1"
q2_owht="1"
if [[ "$VARIANT" == "plain_int2" ]]; then
  model="$BASE_MODEL"
  no_hadamard="0"
  q2_owht="0"
fi

run_one() {
  local mode="$1"
  local prompt="$2"
  local label="${mode}_p${prompt}_${VARIANT}"
  local timeout
  timeout="$(timeout_for "$mode" "$prompt")"

  local -a hp_env=(
    LLAMA_KV_HP_SINK=0
    LLAMA_KV_HP_RECENT=0
    LLAMA_KV_HP_PREFILL_ATTENTION=0
  )
  local -a mixed_env=()
  case "$mode" in
    mixed_fused)
      hp_env=(LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1)
      ;;
    mixed_vec)
      hp_env=(LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1)
      mixed_env=(LLAMA_KV_MIXED_VEC_MAIN=1)
      cache_k="oscar2"
      cache_v="oscar2"
      q2_owht="0"
      ;;
    pure)
      ;;
  esac
  if [[ "$mode" != "mixed_vec" ]]; then
    cache_k="q2_0"
    cache_v="q2_0"
    if [[ "$VARIANT" == "oscar_int2" ]]; then
      q2_owht="1"
    else
      q2_owht="0"
    fi
  fi

  local cmd=(
    env VRAM_POLL_INTERVAL="$VRAM_POLL_INTERVAL" MAX_PEAK_MIB="$MAX_PEAK_MIB"
    "$ROOT_DIR/scripts/measure_vram.sh" "$OUT_DIR" "$label" --
    env
      LLAMA_KV_Q2_0_OWHT="$q2_owht"
      LLAMA_KV_NO_HADAMARD="$no_hadamard"
      LLAMA_KV_CLIP_RATIO=0
      LLAMA_KV_CLIP_RATIO_K=0
      LLAMA_KV_CLIP_RATIO_V=0
      LLAMA_TURBO_VEC_STREAM_K=0
      "${hp_env[@]}"
      "${mixed_env[@]}"
    "$LLAMA_BENCH"
      -m "$model"
      -p "$prompt"
      -n "$GEN_TOKENS"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      -r "$REPETITIONS"
      --cache-type-k "$cache_k"
      --cache-type-v "$cache_v"
      --output json
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN %s:' "$label"
    printf ' %q' timeout "${timeout}s" "${cmd[@]}"
    printf '\n'
    return 0
  fi

  echo "Running $label"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=INT --kill-after=30s "${timeout}s" "${cmd[@]}" > "$OUT_DIR/$label.measure_stdout.txt"
  else
    "${cmd[@]}" > "$OUT_DIR/$label.measure_stdout.txt"
  fi
  local rc=$?
  set -e
  [[ -f "$OUT_DIR/$label.stdout.txt" ]] && mv "$OUT_DIR/$label.stdout.txt" "$OUT_DIR/$label.json"
  if (( POST_CASE_COOLDOWN_SEC > 0 )); then
    sleep "$POST_CASE_COOLDOWN_SEC"
  fi
  return "$rc"
}

if [[ "$DRY_RUN" != "1" ]]; then
  cat > "$OUT_DIR/config.txt" <<EOF
variant=$VARIANT
mode=$MODE
prompts=${prompts[*]}
include_6144=$INCLUDE_6144
max_peak_mib=$MAX_PEAK_MIB
post_case_cooldown_sec=$POST_CASE_COOLDOWN_SEC
EOF
fi

overall=0
for prompt in "${prompts[@]}"; do
  for mode in "${modes[@]}"; do
    run_one "$mode" "$prompt" || overall=$?
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete; set ACK_MIXED_VEC_RAMP=1 DRY_RUN=0 to execute."
else
  python3 "$ROOT_DIR/scripts/summarize_mixed_vec_int2_ramp.py" "$OUT_DIR"
  echo "Ramp results written under: $OUT_DIR"
  exit "$overall"
fi
