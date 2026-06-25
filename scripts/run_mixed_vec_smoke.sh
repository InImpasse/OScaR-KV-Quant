#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-completion}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/mixed_vec_smoke_$STAMP}"
DRY_RUN="${DRY_RUN:-1}"
ACK_MIXED_VEC_SMOKE="${ACK_MIXED_VEC_SMOKE:-0}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
PROMPT="${PROMPT:-What is 2+2?}"
N_PREDICT="${N_PREDICT:-32}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-120}"

if [[ "$DRY_RUN" != "1" && "$ACK_MIXED_VEC_SMOKE" != "1" ]]; then
  echo "Refusing mixed vec smoke without ACK_MIXED_VEC_SMOKE=1." >&2
  echo "Use DRY_RUN=1 to inspect commands or ACK_MIXED_VEC_SMOKE=1 DRY_RUN=0 to run intentionally." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_COMPLETION" ]] || { echo "llama-completion not found: $LLAMA_COMPLETION" >&2; exit 1; }
  mkdir -p "$OUT_DIR/direct" "$OUT_DIR/quality/raw"
fi

run_direct() {
  local label="$1"
  local model="$2"
  local cache_k="$3"
  local cache_v="$4"
  shift 4
  local -a extra_env=("$@")

  local cmd=(
    env
      LLAMA_KV_HP_SINK=0
      LLAMA_KV_HP_RECENT=0
      LLAMA_KV_HP_PREFILL_ATTENTION=0
      LLAMA_KV_Q2_0_OWHT=0
      LLAMA_KV_NO_HADAMARD=0
      LLAMA_KV_CLIP_RATIO=0
      LLAMA_TURBO_VEC_STREAM_K=0
      "${extra_env[@]}"
    "$LLAMA_COMPLETION"
      -m "$model"
      -c "$CTX_SIZE"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      --cache-type-k "$cache_k"
      --cache-type-v "$cache_v"
      -n "$N_PREDICT"
      -p "$PROMPT"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN direct %s:' "$label"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  set +e
  if command -v timeout >/dev/null 2>&1 && (( CASE_TIMEOUT_SEC > 0 )); then
    timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT_SEC}s" "${cmd[@]}" \
      > "$OUT_DIR/direct/${label}.stdout.txt" \
      2> "$OUT_DIR/direct/${label}.stderr.txt"
  else
    "${cmd[@]}" \
      > "$OUT_DIR/direct/${label}.stdout.txt" \
      2> "$OUT_DIR/direct/${label}.stderr.txt"
  fi
  local rc=$?
  set -e
  printf '%s rc=%s stdout=%s\n' "$label" "$rc" "$(tr '\n' ' ' < "$OUT_DIR/direct/${label}.stdout.txt" | head -c 120)" \
    >> "$OUT_DIR/direct/summary.txt"
}

run_quality() {
  local variants="$1"
  local extra_env="${2:-}"

  local args=(
    python3 "$ROOT_DIR/scripts/run_gpqa_gsm8k_cli_eval.py"
      --out-dir "$OUT_DIR/quality"
      --variants "$variants"
      --datasets gpqa,gsm8k
      --gpqa-n-cases 3
      --gsm8k-n-cases 3
      --ctx-size "$CTX_SIZE"
      --case-timeout "$CASE_TIMEOUT_SEC"
  )
  if [[ -n "$extra_env" ]]; then
    args+=(--extra-env "$extra_env")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  else
    args+=(--real --ack-eval)
  fi

  printf 'quality variants=%s extra_env=%s\n' "$variants" "$extra_env" >> "${OUT_DIR:-/tmp}/paths.txt" 2>/dev/null || true
  "${args[@]}"
}

OSCAR_MODEL="$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf"
BASE_MODEL="$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf"

if [[ "$DRY_RUN" != "1" ]]; then
  : > "$OUT_DIR/direct/summary.txt"
  cat > "$OUT_DIR/config.txt" <<EOF
ctx_size=$CTX_SIZE
prompt=$PROMPT
n_predict=$N_PREDICT
case_timeout_sec=$CASE_TIMEOUT_SEC
paths=default_mixed,mixed_vec,pure_q2_vec
EOF
fi

echo "=== mixed vec smoke: default fused mixed (oscar_int2_mixed) ==="
run_direct "default_mixed" "$OSCAR_MODEL" "q2_0" "q2_0" \
  LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_Q2_0_OWHT=1 \
  LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1

echo "=== mixed vec smoke: experimental mixed vec (oscar2_int2_mixed_vec) ==="
run_direct "mixed_vec" "$OSCAR_MODEL" "oscar2" "oscar2" \
  LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0 \
  LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1 \
  LLAMA_KV_MIXED_VEC_MAIN=1

echo "=== mixed vec smoke: pure q2 vec controls ==="
run_direct "plain_int2" "$BASE_MODEL" "q2_0" "q2_0"
run_direct "oscar_int2" "$OSCAR_MODEL" "q2_0" "q2_0" \
  LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_Q2_0_OWHT=1

echo "=== mixed vec smoke: GPQA/GSM8K 3-case quality ==="
run_quality "oscar_int2_mixed,oscar2_int2_mixed_vec,plain_int2,oscar_int2,baseline_bf16"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete; set ACK_MIXED_VEC_SMOKE=1 DRY_RUN=0 to execute."
else
  python3 "$ROOT_DIR/scripts/summarize_gpqa_gsm8k_kv_eval.py" "$OUT_DIR/quality"
  python3 "$ROOT_DIR/scripts/summarize_mixed_vec_smoke.py" "$OUT_DIR"
  echo
  echo "Mixed vec smoke results written under: $OUT_DIR"
fi
