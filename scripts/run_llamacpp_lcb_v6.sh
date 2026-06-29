#!/usr/bin/env bash
# Run LiveCodeBench v6 against llama.cpp/llama-server OpenAI-compatible API.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-server}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$RUNS_DIR/llamacpp_lcb_v6_$STAMP}"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
OSCAR_MODEL="${OSCAR_MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
VARIANTS="${VARIANTS:-baseline_bf16,oscar_int4,plain_int4}"
LCB_ROOT="${LIVE_CODE_BENCH_ROOT:-$ROOT_DIR/third_party/LiveCodeBench}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
PORT_BASE="${PORT_BASE:-8240}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-180}"
POST_READY_SLEEP="${POST_READY_SLEEP:-3}"
SERVER_PARALLEL="${SERVER_PARALLEL:-1}"

LCB_MODEL_NAME="${LCB_MODEL_NAME:-granite-4.0-1b-base}"
LCB_MODEL_REGISTRY_KEY="${LCB_MODEL_REGISTRY_KEY:-auto}"
LCB_RELEASE="${LCB_RELEASE:-release_v6}"
LCB_N="${LCB_N:-1}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0.2}"
LCB_TOP_P="${LCB_TOP_P:-0.95}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-2000}"
LCB_MULTIPROCESS="${LCB_MULTIPROCESS:-1}"
LCB_NUM_PROCESS_EVALUATE="${LCB_NUM_PROCESS_EVALUATE:-1}"
LCB_TIMEOUT="${LCB_TIMEOUT:-6}"
LCB_EXTRA_ARGS=(${LCB_EXTRA_ARGS:-})

DRY_RUN="${DRY_RUN:-1}"
ACK_EVAL="${ACK_EVAL:-0}"
ALLOW_CODE_EXEC="${ALLOW_CODE_EXEC:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/run_llamacpp_lcb_v6.sh [env overrides]

Runs LiveCodeBench v6 code-generation eval through llama-server.

Required for real execution:
  DRY_RUN=0 ACK_EVAL=1 ALLOW_CODE_EXEC=1

Useful env:
  VARIANTS=baseline_bf16,oscar_int4,plain_int4
  LIVE_CODE_BENCH_ROOT=third_party/LiveCodeBench
  LCB_MODEL_NAME=granite-4.0-1b-base
  LCB_MODEL_REGISTRY_KEY=auto maps LCB_MODEL_NAME to an available OpenAI-compatible adapter key.
  LCB_RELEASE=release_v6
  LCB_N=1
  SKIP_COMPLETED=1 skips variants with .done marker files.
  OUT_DIR=runs/...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$DRY_RUN" != "1" && "$ACK_EVAL" != "1" ]]; then
  echo "Refusing real LCB eval without ACK_EVAL=1." >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && "$ALLOW_CODE_EXEC" != "1" ]]; then
  echo "Refusing real LCB eval without ALLOW_CODE_EXEC=1; LiveCodeBench executes generated code." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -x "$LLAMA_SERVER" ]] || { echo "llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
  [[ -f "$BASE_MODEL" ]] || { echo "BASE_MODEL not found: $BASE_MODEL" >&2; exit 1; }
  [[ -f "$OSCAR_MODEL" ]] || { echo "OSCAR_MODEL not found: $OSCAR_MODEL" >&2; exit 1; }
  [[ -d "$LCB_ROOT/lcb_runner" ]] || { echo "LiveCodeBench checkout not found: $LCB_ROOT/lcb_runner" >&2; exit 1; }
fi

resolve_lcb_model_registry_key() {
  "$PY" - "$LCB_ROOT" "$LCB_MODEL_NAME" "$LCB_MODEL_REGISTRY_KEY" <<'PY'
import sys

lcb_root, model_name, requested = sys.argv[1:4]
sys.path.insert(0, lcb_root)

try:
    from lcb_runner.lm_styles import LanguageModelStore
except Exception as exc:
    print(f"ERROR: could not import LiveCodeBench LanguageModelStore: {exc}", file=sys.stderr)
    raise SystemExit(2)

keys = list(LanguageModelStore.keys())
if model_name in LanguageModelStore:
    print(model_name)
    raise SystemExit(0)

if requested != "auto":
    if requested in LanguageModelStore:
        print(requested)
        raise SystemExit(0)
    print(f"ERROR: LCB_MODEL_REGISTRY_KEY={requested!r} is not registered by LiveCodeBench.", file=sys.stderr)
    print("Available model keys:", file=sys.stderr)
    for key in sorted(keys):
        print(f"  {key}", file=sys.stderr)
    raise SystemExit(2)

preferred = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
]
for key in preferred:
    if key in LanguageModelStore:
        print(key)
        raise SystemExit(0)

scored = []
for key in keys:
    low = key.lower()
    score = 0
    if "gpt" in low or "openai" in low:
        score += 10
    if "turbo" in low or "4o" in low:
        score += 3
    if "chat" in low or "instruct" in low:
        score += 1
    if score:
        scored.append((score, key))

if scored:
    print(sorted(scored, key=lambda item: (-item[0], item[1]))[0][1])
    raise SystemExit(0)

print(f"ERROR: {model_name!r} is not registered and no OpenAI-like fallback key was found.", file=sys.stderr)
print("Available model keys:", file=sys.stderr)
for key in sorted(keys):
    print(f"  {key}", file=sys.stderr)
raise SystemExit(2)
PY
}

LCB_MODEL_REGISTRY_EFFECTIVE="$LCB_MODEL_NAME"
if [[ "$DRY_RUN" != "1" ]]; then
  LCB_MODEL_REGISTRY_EFFECTIVE="$(resolve_lcb_model_registry_key)"
  if [[ "$LCB_MODEL_REGISTRY_EFFECTIVE" != "$LCB_MODEL_NAME" ]]; then
    echo "LCB model adapter: $LCB_MODEL_NAME -> $LCB_MODEL_REGISTRY_EFFECTIVE"
  fi
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/raw"
cat > "$OUT_DIR/config.txt" <<EOF
variants=$VARIANTS
lcb_root=$LCB_ROOT
lcb_model_name=$LCB_MODEL_NAME
lcb_model_registry_key=$LCB_MODEL_REGISTRY_EFFECTIVE
lcb_release=$LCB_RELEASE
lcb_n=$LCB_N
lcb_temperature=$LCB_TEMPERATURE
lcb_top_p=$LCB_TOP_P
lcb_max_tokens=$LCB_MAX_TOKENS
ctx_size=$CTX_SIZE
dry_run=$DRY_RUN
skip_completed=$SKIP_COMPLETED
EOF

server_pid=""
cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

probe_ready() {
  local port="$1"
  curl --noproxy '*' --max-time 10 -sf \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"llama\",\"prompt\":\"ready\",\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:${port}/v1/completions" >/dev/null 2>&1
}

wait_for_server() {
  local port="$1"
  local deadline=$(( $(date +%s) + SERVER_TIMEOUT_SEC ))
  while true; do
    if probe_ready "$port"; then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "server did not become completion-ready on port $port" >&2
      return 1
    fi
    sleep 2
  done
}

enabled() {
  local needle="$1"
  [[ ",$VARIANTS," == *",$needle,"* || "$VARIANTS" == "all" ]]
}

run_variant() {
  local label="$1"
  local model="$2"
  local cache_k="$3"
  local cache_v="$4"
  local no_hadamard="$5"
  local clip_ratio="$6"
  local port="$7"
  local q2_owht="0"
  if [[ "$cache_k/$cache_v" == "q2_0/q2_0" && "$no_hadamard" == "1" ]]; then
    q2_owht="1"
  fi
  local run_dir="$OUT_DIR/raw/$label"
  local done_marker="$run_dir/.done"
  mkdir -p "$run_dir"
  if [[ "$DRY_RUN" != "1" && "$SKIP_COMPLETED" == "1" && -f "$done_marker" ]]; then
    echo "Skipping completed LCB $label"
    return 0
  fi

  local server_cmd=(
    env
      CUDA_VISIBLE_DEVICES="$GPU"
      LLAMA_KV_HP_SINK=0
      LLAMA_KV_HP_RECENT=0
      LLAMA_KV_Q2_0_OWHT="$q2_owht"
      LLAMA_KV_NO_HADAMARD="$no_hadamard"
      LLAMA_KV_CLIP_RATIO="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_K="$clip_ratio"
      LLAMA_KV_CLIP_RATIO_V="$clip_ratio"
    "$LLAMA_SERVER"
      -m "$model"
      -c "$CTX_SIZE"
      -ngl "$N_GPU_LAYERS"
      -fa "$FLASH_ATTN"
      -ctk "$cache_k"
      -ctv "$cache_v"
      --host 127.0.0.1
      --port "$port"
      --no-webui
      --chat-template granite
      -np "$SERVER_PARALLEL"
      --cache-ram 0
      --no-cache-prompt
      --ctx-checkpoints 0
      --no-warmup
  )
  local lcb_cmd=(
    "$PY" -m lcb_runner.runner.main
      --model "$LCB_MODEL_REGISTRY_EFFECTIVE"
      --scenario codegeneration
      --release_version "$LCB_RELEASE"
      --evaluate
      --n "$LCB_N"
      --temperature "$LCB_TEMPERATURE"
      --top_p "$LCB_TOP_P"
      --max_tokens "$LCB_MAX_TOKENS"
      --multiprocess "$LCB_MULTIPROCESS"
      --num_process_evaluate "$LCB_NUM_PROCESS_EVALUATE"
      --timeout "$LCB_TIMEOUT"
      "${LCB_EXTRA_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN server %s:' "$label"; printf ' %q' "${server_cmd[@]}"; printf '\n'
    printf 'DRY_RUN lcb %s:' "$label"
    printf ' OPENAI_API_KEY=EMPTY OPENAI_KEY=EMPTY OPENAI_BASE_URL=%q' "http://127.0.0.1:${port}/v1"
    printf ' PYTHONPATH=%q' "$LCB_ROOT:${PYTHONPATH:-}"
    printf ' %q' "${lcb_cmd[@]}"
    printf '\n'
    return 0
  fi

  echo "Starting llama-server for $label on port $port"
  "${server_cmd[@]}" > "$OUT_DIR/logs/${label}.server.log" 2>&1 &
  server_pid="$!"
  wait_for_server "$port"
  sleep "$POST_READY_SLEEP"

  (
    cd "$LCB_ROOT"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    export OPENAI_KEY="${OPENAI_KEY:-EMPTY}"
    export OPENAI_BASE_URL="http://127.0.0.1:${port}/v1"
    export LCB_USE_COMPLETIONS="${LCB_USE_COMPLETIONS:-1}"
    export PYTHONPATH="$LCB_ROOT:${PYTHONPATH:-}"
    "${lcb_cmd[@]}"
  ) 2>&1 | tee "$OUT_DIR/logs/${label}.lcb.log"

  mkdir -p "$run_dir/lcb_output"
  if [[ -d "$LCB_ROOT/output" ]]; then
    cp -a "$LCB_ROOT/output/." "$run_dir/lcb_output/" || true
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$done_marker"
  cleanup
  server_pid=""
}

i=0
if enabled baseline_bf16; then
  run_variant baseline_bf16 "$BASE_MODEL" bf16 bf16 0 0 "$((PORT_BASE + i))"
  i=$((i + 1))
fi
if enabled oscar_int4; then
  run_variant oscar_int4 "$OSCAR_MODEL" q4_0 q4_0 1 0.96 "$((PORT_BASE + i))"
  i=$((i + 1))
fi
if enabled plain_int4; then
  run_variant plain_int4 "$BASE_MODEL" q4_0 q4_0 0 0 "$((PORT_BASE + i))"
  i=$((i + 1))
fi
if enabled oscar_int2; then
  run_variant oscar_int2 "$OSCAR_MODEL" q2_0 q2_0 1 0 "$((PORT_BASE + i))"
  i=$((i + 1))
fi
if enabled plain_int2; then
  run_variant plain_int2 "$BASE_MODEL" q2_0 q2_0 0 0 "$((PORT_BASE + i))"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no results written."
else
  echo "LCB outputs copied under: $OUT_DIR/raw/<variant>/lcb_output"
fi
