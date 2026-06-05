#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUT_DIR LABEL -- COMMAND [ARGS...]" >&2
  exit 2
fi

OUT_DIR="$1"
LABEL="$2"
shift 2

if [[ "${1:-}" != "--" ]]; then
  echo "expected -- before command" >&2
  exit 2
fi
shift

mkdir -p "$OUT_DIR"

STDOUT_FILE="$OUT_DIR/$LABEL.stdout.txt"
STDERR_FILE="$OUT_DIR/$LABEL.stderr.txt"
METRICS_FILE="$OUT_DIR/$LABEL.metrics.tsv"
SUMMARY_FILE="$OUT_DIR/$LABEL.summary.txt"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot sample VRAM. Set MEASURE_VRAM=0 for scripts that support it, or fix NVIDIA tooling." >&2
  exit 1
fi

set +e
baseline_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)"
baseline_status=$?
set -e
baseline="$(printf '%s\n' "$baseline_raw" | head -1 | tr -d ' ')"
if [[ ! "$baseline" =~ ^[0-9]+$ ]]; then
  echo "nvidia-smi failed or returned non-numeric memory usage; cannot sample VRAM. Run ./scripts/check_kv_env.sh." >&2
  echo "nvidia-smi exit_code=$baseline_status output=$baseline_raw" >&2
  exit 1
fi
peak="$baseline"

start_ns="$(date +%s%N)"
"$@" >"$STDOUT_FILE" 2>"$STDERR_FILE" &
cmd_pid="$!"

{
  echo -e "timestamp_ms\tmemory_used_mib"
  while kill -0 "$cmd_pid" 2>/dev/null; do
    now_ns="$(date +%s%N)"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak )); then
      peak="$mem"
    fi
    echo -e "$(((now_ns - start_ns) / 1000000))\t$mem"
    sleep "${VRAM_POLL_INTERVAL:-0.2}"
  done
} >"$METRICS_FILE"

set +e
wait "$cmd_pid"
exit_code="$?"
set -e

end_ns="$(date +%s%N)"
duration_ms="$(((end_ns - start_ns) / 1000000))"

if [[ "${REQUIRE_PPL_OUTPUT:-0}" == "1" ]] && [[ "$exit_code" -eq 0 ]] && \
   ! grep -Eq '(Final estimate:[[:space:]]*)?PPL[[:space:]]*=' "$STDOUT_FILE" "$STDERR_FILE"; then
  exit_code=2
fi

{
  echo "label=$LABEL"
  echo "exit_code=$exit_code"
  echo "duration_ms=$duration_ms"
  echo "baseline_mib=$baseline"
  echo "peak_mib=$peak"
  echo "delta_mib=$((peak - baseline))"
  echo "stdout=$STDOUT_FILE"
  echo "stderr=$STDERR_FILE"
  echo "metrics=$METRICS_FILE"
} >"$SUMMARY_FILE"

cat "$SUMMARY_FILE"
exit "$exit_code"
