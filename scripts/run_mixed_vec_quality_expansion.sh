#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/mixed_vec_quality_expansion_$STAMP}"
DRY_RUN="${DRY_RUN:-1}"
ACK_QUALITY_EXPANSION="${ACK_QUALITY_EXPANSION:-0}"
RAMP_DIR="${RAMP_DIR:-$RUNS_DIR/mixed_vec_int2_ramp_current}"
MIN_8K_PP="${MIN_8K_PP:-310}"
CTX_SIZE="${CTX_SIZE:-8192}"

if [[ "$DRY_RUN" != "1" && "$ACK_QUALITY_EXPANSION" != "1" ]]; then
  echo "Refusing quality expansion without ACK_QUALITY_EXPANSION=1." >&2
  exit 2
fi

gate_ok=0
if [[ -f "$RAMP_DIR/ramp.csv" ]]; then
  gate_ok="$(python3 - "$RAMP_DIR/ramp.csv" "$MIN_8K_PP" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
threshold = float(sys.argv[2])
rows = list(csv.DictReader(path.open()))
best = 0.0
for row in rows:
    if not row["prompt"].endswith("8192") and row["prompt"] != "8192":
        continue
    if row["status"] != "ok":
        continue
    try:
        best = max(best, float(row["pp_tps"]))
    except ValueError:
        pass
print(1 if best >= threshold else 0)
PY
)"
fi

if [[ "$gate_ok" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "Refusing quality expansion: 8K pp gate not met in $RAMP_DIR (need >= $MIN_8K_PP tok/s)." >&2
  exit 3
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT_DIR"
fi

run_ppl() {
  if [[ -z "${CORPUS:-}" ]]; then
    echo "Skipping PPL matrix: CORPUS unset."
    return 0
  fi
  local cmd=(
    env OUT_DIR="$OUT_DIR/ppl" DRY_RUN="$DRY_RUN"
    CONTEXTS="medium:2048,long:8192"
    KV_MODES="bf16,q2_0,q2_0_hp"
    MODELS="granite:$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf"
    "$ROOT_DIR/scripts/run_kv_ppl_matrix.sh"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN ppl:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    ACK_PPL_MATRIX=1 DRY_RUN=0 OUT_DIR="$OUT_DIR/ppl" \
      CONTEXTS="medium:2048,long:8192" \
      KV_MODES="bf16,q2_0,q2_0_hp" \
      MODELS="granite:$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf" \
      "$ROOT_DIR/scripts/run_kv_ppl_matrix.sh"
  fi
}

run_long_ctx_eval() {
  local args=(
    python3 "$ROOT_DIR/scripts/run_gpqa_gsm8k_cli_eval.py"
      --out-dir "$OUT_DIR/long_ctx"
      --variants baseline_bf16,oscar_int2,oscar_int2_mixed,oscar2_int2_mixed_vec
      --datasets gpqa,gsm8k
      --gpqa-n-cases 10
      --gsm8k-n-cases 10
      --ctx-size "$CTX_SIZE"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  else
    args+=(--real --ack-eval)
  fi
  "${args[@]}"
}

echo "=== quality expansion: PPL matrix ==="
run_ppl
echo "=== quality expansion: long-context GPQA/GSM8K ==="
run_long_ctx_eval

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete; requires 8K pp >= $MIN_8K_PP in $RAMP_DIR before real run."
  echo "Set ACK_QUALITY_EXPANSION=1 DRY_RUN=0 to execute after gate passes."
else
  python3 "$ROOT_DIR/scripts/summarize_gpqa_gsm8k_kv_eval.py" "$OUT_DIR/long_ctx" || true
  if [[ -d "$OUT_DIR/ppl/raw" ]]; then
    python3 "$ROOT_DIR/scripts/summarize_kv_ppl.py" "$OUT_DIR/ppl" || true
  fi
  echo "Quality expansion results written under: $OUT_DIR"
fi
