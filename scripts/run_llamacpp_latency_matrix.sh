#!/usr/bin/env bash
# Measure llama-server streaming decode latency for the llama.cpp preset matrix.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=llamacpp_matrix_common.sh
source "$ROOT_DIR/scripts/llamacpp_matrix_common.sh"

LLAMA_SERVER="${LLAMA_SERVER:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-server}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$RUNS_DIR/llamacpp_latency_matrix_$STAMP}"
PRESETS="${PRESETS:-short,medium,long,16k,32k}"
MODES="${MODES:-bf16,oscar-int2,int2}"
GEN_TOKENS="${GEN_TOKENS:-64}"
CTX_EXTRA_TOKENS="${CTX_EXTRA_TOKENS:-128}"
PORT="${PORT:-8033}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-180}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-900}"
DRY_RUN="${DRY_RUN:-1}"
CUDA_GRAPHS_MODE="${CUDA_GRAPHS_MODE:-auto}"
CUDA_GRAPH_OPT="${CUDA_GRAPH_OPT:-0}"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"

usage() {
  cat <<'EOF'
Usage: scripts/run_llamacpp_latency_matrix.sh [options]

Measures Decode first and P95 through llama-server streaming. Defaults to dry-run.

Options:
  --out-root DIR       Output parent directory
  --presets LIST      Comma list: short,medium,long,16k,32k
  --modes LIST        Comma list: bf16,int2,oscar-int2
  --gen-tokens N      Tokens to stream for latency sampling
  --port PORT         llama-server port
  --real              Execute benchmarks
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --presets) PRESETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --real) DRY_RUN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/health"
  local deadline=$(( $(date +%s) + SERVER_TIMEOUT_SEC ))
  while true; do
    if python3 - "$url" >/dev/null 2>&1 <<'PY'
import sys
from urllib import request
try:
    with request.urlopen(sys.argv[1], timeout=2) as r:
        raise SystemExit(0 if r.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "server did not become ready: $url" >&2
      return 1
    fi
    sleep 1
  done
}

server_pid=""
cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mode_meta() {
  case "$1" in
    bf16) printf '%s\t%s\t%s\t%s\t%s\n' "$BASE_MODEL" bf16 bf16 0 0 ;;
    int2|plain_int2) printf '%s\t%s\t%s\t%s\t%s\n' "$BASE_MODEL" q2_0 q2_0 0 0 ;;
    oscar-int2|oscar_int2) printf '%s\t%s\t%s\t%s\t%s\n' "$OSCAR_MODEL" q2_0 q2_0 1 0 ;;
    *) echo "Unsupported mode '$1'." >&2; return 2 ;;
  esac
}

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_SERVER" ]] || { echo "llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
  [[ -f "$BASE_MODEL" ]] || { echo "BASE_MODEL not found: $BASE_MODEL" >&2; exit 1; }
  [[ -f "$OSCAR_MODEL" ]] || { echo "OSCAR_MODEL not found: $OSCAR_MODEL" >&2; exit 1; }
fi

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/manual"

IFS=',' read -r -a mode_list <<< "$MODES"
IFS=',' read -r -a preset_list <<< "$PRESETS"
for raw_mode in "${mode_list[@]}"; do
  mode="$(echo "$raw_mode" | xargs)"
  [[ -n "$mode" ]] || continue
  read -r model cache_k cache_v no_hadamard clip_ratio < <(mode_meta "$mode")
  q2_owht=0
  [[ "$cache_k/$cache_v" == "q2_0/q2_0" && "$no_hadamard" == "1" ]] && q2_owht=1

  graph_env=()
  case "$CUDA_GRAPHS_MODE" in
    on) graph_env+=(GGML_CUDA_DISABLE_GRAPHS=) ;;
    off) graph_env+=(GGML_CUDA_DISABLE_GRAPHS=1) ;;
    auto) ;;
    *) echo "Unsupported CUDA_GRAPHS_MODE=$CUDA_GRAPHS_MODE" >&2; exit 2 ;;
  esac

  server_cmd=(
    env
      "${graph_env[@]}"
      GGML_CUDA_GRAPH_OPT="$CUDA_GRAPH_OPT"
      LLAMA_KV_HP_SINK=0
      LLAMA_KV_HP_RECENT=0
      LLAMA_KV_Q2_0_OWHT="$q2_owht"
      LLAMA_KV_NO_HADAMARD="$no_hadamard"
      LLAMA_KV_CLIP_RATIO="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_K="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_V="$clip_ratio"
    "$LLAMA_SERVER"
      -m "$model"
      -c "$((32768 + CTX_EXTRA_TOKENS))"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      -ctk "$cache_k"
      -ctv "$cache_v"
      --host 127.0.0.1
      --port "$PORT"
      --no-webui
      --chat-template granite
      -np 1
      --cache-ram 0
      --no-cache-prompt
      --ctx-checkpoints 0
      --no-warmup
      --log-disable
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN server %s:' "$mode"
    printf ' %q' "${server_cmd[@]}"
    printf '\n'
  else
    echo "[run_llamacpp_latency_matrix] starting server mode=$mode"
    "${server_cmd[@]}" > "$OUT_ROOT/logs/${mode}.server.log" 2>&1 &
    server_pid="$!"
    wait_for_server
  fi

  for raw_preset in "${preset_list[@]}"; do
    preset="$(echo "$raw_preset" | xargs)"
    [[ -n "$preset" ]] || continue
    tokens="$(llamacpp_preset_tokens "$preset")"
    out_csv="$OUT_ROOT/manual/${preset}_${mode}.csv"
    cmd=(
      python3 "$ROOT_DIR/scripts/measure_llamacpp_decode_latency.py"
        --server "http://127.0.0.1:${PORT}"
        --preset "$preset"
        --mode "$mode"
        --max-tokens "$GEN_TOKENS"
        --timeout "$REQUEST_TIMEOUT_SEC"
        --out "$out_csv"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY_RUN latency preset=%s mode=%s tokens=%s:' "$preset" "$mode" "$tokens"
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      echo "[run_llamacpp_latency_matrix] preset=$preset mode=$mode"
      "${cmd[@]}" > "$OUT_ROOT/logs/${preset}_${mode}.latency.log" 2>&1
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    cleanup
    server_pid=""
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  python3 - "$OUT_ROOT/manual_metrics.csv" "$OUT_ROOT"/manual/*.csv <<'PY'
import csv
import sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for name in sys.argv[2:]:
    with Path(name).open(newline="") as f:
        rows.extend(csv.DictReader(f))
fields = ["preset", "mode", "prefill_tokens", "decode_first_tok_s", "decode_steady_median_tok_s", "decode_steady_p95_tok_s"]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
print(f"wrote {out}")
PY
  echo "[run_llamacpp_latency_matrix] results -> $OUT_ROOT"
else
  echo "[run_llamacpp_latency_matrix] dry run complete; pass --real or DRY_RUN=0 to execute."
fi
