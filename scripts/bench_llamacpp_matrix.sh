#!/usr/bin/env bash
# Run llama.cpp BF16 / INT2 / INT4 variants across SGLang-style context presets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=llamacpp_matrix_common.sh
source "$ROOT_DIR/scripts/llamacpp_matrix_common.sh"

RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$RUNS_DIR/llamacpp_bench_matrix_$STAMP}"
PRESETS="${PRESETS:-short,medium,long,16k,32k}"
MODES="${MODES:-bf16,oscar-int4,int4}"
GEN_TOKENS="${GEN_TOKENS:-1}"
REPETITIONS="${REPETITIONS:-1}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-180}"
DRY_RUN="${DRY_RUN:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ACK_HEAVY_32K="${ACK_HEAVY_32K:-0}"
ACK_Q2_32K_NOGO="${ACK_Q2_32K_NOGO:-0}"
ACK_Q2_RAMP_GATE_HOLD="${ACK_Q2_RAMP_GATE_HOLD:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/bench_llamacpp_matrix.sh [options]

Runs scripts/bench_32k_llamacpp_kv.sh for each preset using SGLang-compatible
preset names. Defaults to dry-run.

Options:
  --out-root DIR       Output parent directory
  --presets LIST      Comma list: short,medium,long,16k,32k
  --modes LIST        Comma list: bf16,int2,oscar-int2,int4,oscar-int4
  --gen-tokens N      llama-bench -n value
  --case-timeout SEC  Per-case timeout
  --real              Execute benchmarks (same as DRY_RUN=0)
  --skip-existing     Skip preset directory if summary.csv already exists
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --presets) PRESETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --case-timeout) CASE_TIMEOUT_SEC="$2"; shift 2 ;;
    --real) DRY_RUN=0; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

CASES="$(llamacpp_cases_from_modes "$MODES")"
mkdir -p "$OUT_ROOT"
manifest="$OUT_ROOT/matrix_manifest.jsonl"
: > "$manifest"

IFS=',' read -r -a preset_list <<< "$PRESETS"
for preset in "${preset_list[@]}"; do
  preset="$(echo "$preset" | xargs)"
  [[ -n "$preset" ]] || continue
  prompt_tokens="$(llamacpp_preset_tokens "$preset")"
  preset_dir="$OUT_ROOT/$preset"
  if [[ "$SKIP_EXISTING" == "1" && -f "$preset_dir/summary.csv" ]]; then
    echo "[bench_llamacpp_matrix] skip preset=$preset: existing summary.csv"
    printf '{"preset":"%s","prompt_tokens":%s,"run_dir":"%s","skipped":true}\n' \
      "$preset" "$prompt_tokens" "$preset_dir" >> "$manifest"
    continue
  fi

  echo "[bench_llamacpp_matrix] preset=$preset prompt=$prompt_tokens cases=$CASES"
  OUT_DIR="$preset_dir" \
  PROMPT_TOKENS="$prompt_tokens" \
  CASES="$CASES" \
  GEN_TOKENS="$GEN_TOKENS" \
  REPETITIONS="$REPETITIONS" \
  CASE_TIMEOUT_SEC="$CASE_TIMEOUT_SEC" \
  DRY_RUN="$DRY_RUN" \
  RUN_PREFLIGHT="$RUN_PREFLIGHT" \
  ACK_HEAVY_32K="$ACK_HEAVY_32K" \
  ACK_Q2_32K_NOGO="$ACK_Q2_32K_NOGO" \
  ACK_Q2_RAMP_GATE_HOLD="$ACK_Q2_RAMP_GATE_HOLD" \
    "$ROOT_DIR/scripts/bench_32k_llamacpp_kv.sh"

  printf '{"preset":"%s","prompt_tokens":%s,"run_dir":"%s","skipped":false}\n' \
    "$preset" "$prompt_tokens" "$preset_dir" >> "$manifest"
done

if [[ "$DRY_RUN" == "0" ]]; then
  python3 "$ROOT_DIR/scripts/summarize_llamacpp_matrix.py" "$OUT_ROOT"
  echo "[bench_llamacpp_matrix] results -> $OUT_ROOT"
else
  echo "[bench_llamacpp_matrix] dry run complete; pass --real or DRY_RUN=0 to execute."
fi
