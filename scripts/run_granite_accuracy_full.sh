#!/usr/bin/env bash
# Full Granite 4.0 1B accuracy harness for BF16, OSCAR INT4, and plain INT4.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/granite_accuracy_full_$STAMP}"

LLAMA_SERVER="${LLAMA_SERVER:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-server}"
LLAMA_EVAL="${LLAMA_EVAL:-$ROOT_DIR/third_party/OSCAR/examples/llama-eval/llama-eval.py}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
LCB_ROOT="${LIVE_CODE_BENCH_ROOT:-$ROOT_DIR/third_party/LiveCodeBench}"

VARIANTS="${VARIANTS:-baseline_bf16,oscar_int4,plain_int4}"
NON_LCB_DATASETS="${NON_LCB_DATASETS:-gpqa,gsm8k,math500,humaneval,aime2025}"
RUN_LCB="${RUN_LCB:-1}"

CTX_SIZE="${CTX_SIZE:-4096}"
PORT="${PORT:-8033}"
LCB_PORT_BASE="${LCB_PORT_BASE:-8240}"
THREADS="${THREADS:-$(nproc)}"
SERVER_PARALLEL="${SERVER_PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-0}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-180}"
RESUME="${RESUME:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

GPQA_N_CASES="${GPQA_N_CASES:-198}"
GSM8K_N_CASES="${GSM8K_N_CASES:-200}"
MATH500_N_CASES="${MATH500_N_CASES:-500}"
HUMANEVAL_N_CASES="${HUMANEVAL_N_CASES:-164}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-5}"
AIME25_N_CASES="${AIME25_N_CASES:-30}"

DRY_RUN="${DRY_RUN:-1}"
ACK_EVAL="${ACK_EVAL:-0}"
ALLOW_HUMANEVAL_EXEC="${ALLOW_HUMANEVAL_EXEC:-0}"
ALLOW_CODE_EXEC="${ALLOW_CODE_EXEC:-0}"
CHECK_DATASETS="${CHECK_DATASETS:-1}"
MIN_VRAM_GB="${MIN_VRAM_GB:-40}"
MIN_RAM_GB="${MIN_RAM_GB:-120}"
MIN_CPU_CORES="${MIN_CPU_CORES:-48}"

usage() {
  cat <<'EOF'
Usage: scripts/run_granite_accuracy_full.sh [env overrides]

Default plan:
  Variants: baseline_bf16, oscar_int4, plain_int4
  Tasks: GPQA, GSM8K, MATH-500, HumanEval, AIME25, LiveCodeBench v6

Real run:
  DRY_RUN=0 ACK_EVAL=1 ALLOW_HUMANEVAL_EXEC=1 ALLOW_CODE_EXEC=1 \
    scripts/run_granite_accuracy_full.sh

Resume:
  OUT_DIR=runs/<existing_run> DRY_RUN=0 ACK_EVAL=1 \
  ALLOW_HUMANEVAL_EXEC=1 ALLOW_CODE_EXEC=1 scripts/run_granite_accuracy_full.sh

Useful overrides:
  OUT_DIR=...
  VARIANTS=baseline_bf16,oscar_int4,plain_int4
  RUN_LCB=0
  THREADS=64
  HUMANEVAL_SAMPLES=5
  CHECK_DATASETS=0
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

die() {
  echo "error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DRY_RUN" != "1" && "$ACK_EVAL" != "1" ]]; then
  die "real eval requires ACK_EVAL=1"
fi
if [[ "$DRY_RUN" != "1" && ",$NON_LCB_DATASETS," == *",humaneval,"* && "$ALLOW_HUMANEVAL_EXEC" != "1" ]]; then
  die "HumanEval executes generated code; set ALLOW_HUMANEVAL_EXEC=1"
fi
if [[ "$DRY_RUN" != "1" && "$RUN_LCB" == "1" && "$ALLOW_CODE_EXEC" != "1" ]]; then
  die "LiveCodeBench executes generated code; set ALLOW_CODE_EXEC=1"
fi

mkdir -p "$OUT_DIR/logs"

check_resources() {
  log "checking local resources"
  local cores
  cores="$(nproc)"
  if (( cores < MIN_CPU_CORES )); then
    log "warning: CPU cores $cores < requested floor $MIN_CPU_CORES"
  fi

  local mem_gb
  mem_gb="$(awk '/MemTotal:/ { printf "%d", $2 / 1024 / 1024 }' /proc/meminfo)"
  if (( mem_gb < MIN_RAM_GB )); then
    log "warning: RAM ${mem_gb}GB < requested floor ${MIN_RAM_GB}GB"
  fi

  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
      > "$OUT_DIR/logs/nvidia-smi.txt" || true
    local max_vram_mb
    max_vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | awk 'BEGIN{m=0} { if ($1 > m) m=$1 } END{ printf "%d", m }')" || max_vram_mb=0
    if (( max_vram_mb == 0 )); then
      log "warning: nvidia-smi did not report GPU memory"
    elif (( max_vram_mb < MIN_VRAM_GB * 1024 )); then
      log "warning: max GPU VRAM ${max_vram_mb}MiB < requested floor $((MIN_VRAM_GB * 1024))MiB"
    fi
  else
    log "warning: nvidia-smi not found; GPU capacity was not checked"
  fi
}

check_files() {
  log "checking binaries and models"
  [[ -x "$LLAMA_SERVER" ]] || die "llama-server not found or not executable: $LLAMA_SERVER"
  [[ -f "$LLAMA_EVAL" ]] || die "llama-eval.py not found: $LLAMA_EVAL"
  [[ -f "$BASE_MODEL" ]] || die "BASE_MODEL not found: $BASE_MODEL"
  [[ -f "$OSCAR_MODEL" ]] || die "OSCAR_MODEL not found: $OSCAR_MODEL"
  if [[ "$RUN_LCB" == "1" && "$DRY_RUN" != "1" ]]; then
    [[ -d "$LCB_ROOT/lcb_runner" ]] || die "LiveCodeBench checkout not found: $LCB_ROOT/lcb_runner"
  fi
}

check_python_deps() {
  log "checking Python dependencies"
  python3 - <<'PY'
import importlib.util
import sys

missing = []
for module in ("requests", "tqdm", "datasets"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
if missing:
    print("missing Python modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  if [[ ",$NON_LCB_DATASETS," == *",humaneval,"* ]]; then
    python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("human_eval") is None:
    print("missing Python module: human_eval", file=sys.stderr)
    raise SystemExit(1)
PY
  fi
}

check_dataset_loads() {
  if [[ "$CHECK_DATASETS" != "1" ]]; then
    log "dataset load preflight disabled"
    return 0
  fi
  log "checking dataset availability; this may download/cache datasets"
  local dataset
  IFS=',' read -ra _datasets <<< "$NON_LCB_DATASETS"
  for dataset in "${_datasets[@]}"; do
    [[ -n "$dataset" ]] || continue
    log "dataset preflight: $dataset"
    python3 - "$LLAMA_EVAL" "$dataset" > "$OUT_DIR/logs/preflight_${dataset}.log" 2>&1 <<'PY' || {
import importlib.util
import sys
from pathlib import Path

eval_py = Path(sys.argv[1])
dataset = sys.argv[2]
spec = importlib.util.spec_from_file_location("llama_eval_preflight", eval_py)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
state = mod.EvalState(dataset_type=dataset, sampling_config={})
state.load_dataset(seed=1234)
state.setup_tasks(n_cases=1, seed=1234)
print(f"{dataset}: ok, available={len(state.dataset)}")
PY
        tail -40 "$OUT_DIR/logs/preflight_${dataset}.log" >&2 || true
        die "dataset preflight failed for $dataset"
      }
  done
}

write_config() {
  cat > "$OUT_DIR/config.txt" <<EOF
root_dir=$ROOT_DIR
base_model=$BASE_MODEL
oscar_model=$OSCAR_MODEL
variants=$VARIANTS
non_lcb_datasets=$NON_LCB_DATASETS
run_lcb=$RUN_LCB
ctx_size=$CTX_SIZE
threads=$THREADS
server_parallel=$SERVER_PARALLEL
dry_run=$DRY_RUN
resume=$RESUME
skip_completed=$SKIP_COMPLETED
gpqa_n_cases=$GPQA_N_CASES
gsm8k_n_cases=$GSM8K_N_CASES
math500_n_cases=$MATH500_N_CASES
humaneval_n_cases=$HUMANEVAL_N_CASES
humaneval_samples=$HUMANEVAL_SAMPLES
aime25_n_cases=$AIME25_N_CASES
EOF
}

run_non_lcb() {
  log "running non-LCB accuracy suite"
  OUT_DIR="$OUT_DIR/non_lcb" \
  LLAMA_SERVER="$LLAMA_SERVER" \
  LLAMA_EVAL="$LLAMA_EVAL" \
  BASE_MODEL="$BASE_MODEL" \
  OSCAR_MODEL="$OSCAR_MODEL" \
  VARIANTS="$VARIANTS" \
  DATASETS="$NON_LCB_DATASETS" \
  CTX_SIZE="$CTX_SIZE" \
  PORT="$PORT" \
  THREADS="$THREADS" \
  SERVER_PARALLEL="$SERVER_PARALLEL" \
  N_GPU_LAYERS="$N_GPU_LAYERS" \
  FLASH_ATTN="$FLASH_ATTN" \
  SERVER_TIMEOUT_SEC="$SERVER_TIMEOUT_SEC" \
  EVAL_TIMEOUT_SEC="$EVAL_TIMEOUT_SEC" \
  RESUME="$RESUME" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  GPQA_N_CASES="$GPQA_N_CASES" \
  GSM8K_N_CASES="$GSM8K_N_CASES" \
  MATH500_N_CASES="$MATH500_N_CASES" \
  HUMANEVAL_N_CASES="$((HUMANEVAL_N_CASES * HUMANEVAL_SAMPLES))" \
  AIME25_N_CASES="$AIME25_N_CASES" \
  DRY_RUN="$DRY_RUN" \
  ACK_EVAL="$ACK_EVAL" \
  ALLOW_HUMANEVAL_EXEC="$ALLOW_HUMANEVAL_EXEC" \
    "$ROOT_DIR/scripts/run_llamacpp_accuracy_suite.sh"
}

run_lcb() {
  if [[ "$RUN_LCB" != "1" ]]; then
    log "skipping LiveCodeBench"
    return 0
  fi
  log "running LiveCodeBench v6"
  OUT_DIR="$OUT_DIR/lcb_v6" \
  LLAMA_SERVER="$LLAMA_SERVER" \
  BASE_MODEL="$BASE_MODEL" \
  OSCAR_MODEL="$OSCAR_MODEL" \
  VARIANTS="$VARIANTS" \
  LIVE_CODE_BENCH_ROOT="$LCB_ROOT" \
  PORT_BASE="$LCB_PORT_BASE" \
  CTX_SIZE="$CTX_SIZE" \
  THREADS="$THREADS" \
  SERVER_PARALLEL="$SERVER_PARALLEL" \
  N_GPU_LAYERS="$N_GPU_LAYERS" \
  FLASH_ATTN="$FLASH_ATTN" \
  SERVER_TIMEOUT_SEC="$SERVER_TIMEOUT_SEC" \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  DRY_RUN="$DRY_RUN" \
  ACK_EVAL="$ACK_EVAL" \
  ALLOW_CODE_EXEC="$ALLOW_CODE_EXEC" \
    "$ROOT_DIR/scripts/run_llamacpp_lcb_v6.sh"
}

summarize() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  python3 "$ROOT_DIR/scripts/summarize_granite_accuracy_full.py" "$OUT_DIR" || true
}

write_config
check_resources
check_files
if [[ "$DRY_RUN" != "1" ]]; then
  check_python_deps
  check_dataset_loads
fi
run_non_lcb
run_lcb
summarize

log "accuracy run directory: $OUT_DIR"
