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
exit_code=130
duration_ms=0
summary_written=0
limit_triggered=0
limit_mib="${MAX_PEAK_MIB:-0}"
if [[ ! "$limit_mib" =~ ^[0-9]+$ ]]; then
  echo "MAX_PEAK_MIB must be a non-negative integer MiB value, got: $limit_mib" >&2
  exit 2
fi

start_ns="$(date +%s%N)"
"$@" >"$STDOUT_FILE" 2>"$STDERR_FILE" &
cmd_pid="$!"

cleanup_child() {
  if kill -0 "$cmd_pid" 2>/dev/null; then
    kill -INT "$cmd_pid" 2>/dev/null || true
    sleep 1
    kill -TERM "$cmd_pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$cmd_pid" 2>/dev/null || true
  fi
}

signal_child_fast() {
  if kill -0 "$cmd_pid" 2>/dev/null; then
    kill -INT "$cmd_pid" 2>/dev/null || true
    sleep 0.2
    kill -TERM "$cmd_pid" 2>/dev/null || true
    sleep 0.2
    kill -KILL "$cmd_pid" 2>/dev/null || true
  fi
}

write_summary() {
  local end_ns
  end_ns="$(date +%s%N)"
  duration_ms="$(((end_ns - start_ns) / 1000000))"
  {
    echo "label=$LABEL"
    echo "exit_code=$exit_code"
    echo "duration_ms=$duration_ms"
    echo "baseline_mib=$baseline"
    echo "peak_mib=$peak"
    echo "delta_mib=$((peak - baseline))"
    echo "max_peak_mib=$limit_mib"
    echo "limit_triggered=$limit_triggered"
    echo "stdout=$STDOUT_FILE"
    echo "stderr=$STDERR_FILE"
    echo "metrics=$METRICS_FILE"
  } >"$SUMMARY_FILE"
  summary_written=1
}

on_exit() {
  local status=$?
  if [[ "$summary_written" != "1" ]]; then
    if kill -0 "$cmd_pid" 2>/dev/null; then
      cleanup_child
      exit_code=124
    else
      set +e
      wait "$cmd_pid"
      exit_code="$?"
      set -e
    fi
    write_summary
  fi
  exit "$status"
}

trap 'exit_code=130; write_summary; signal_child_fast; exit 130' INT
trap 'exit_code=143; write_summary; signal_child_fast; exit 143' TERM
trap on_exit EXIT

{
  echo -e "timestamp_ms\tmemory_used_mib"
  while kill -0 "$cmd_pid" 2>/dev/null; do
    now_ns="$(date +%s%N)"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak )); then
      peak="$mem"
    fi
    echo -e "$(((now_ns - start_ns) / 1000000))\t$mem"
    if (( limit_mib > 0 )) && [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > limit_mib )); then
      limit_triggered=1
      exit_code=137
      echo "MAX_PEAK_MIB exceeded: memory_used_mib=$mem limit_mib=$limit_mib" >>"$STDERR_FILE"
      signal_child_fast
      break
    fi
    sleep "${VRAM_POLL_INTERVAL:-0.2}"
  done
} >"$METRICS_FILE"

set +e
wait "$cmd_pid"
wait_status="$?"
set -e
if [[ "$limit_triggered" == "1" ]]; then
  exit_code=137
else
  exit_code="$wait_status"
fi

trap - INT TERM

if [[ "${REQUIRE_PPL_OUTPUT:-0}" == "1" ]] && [[ "$exit_code" -eq 0 ]] && \
   ! grep -Eq '(Final estimate:[[:space:]]*)?PPL[[:space:]]*=' "$STDOUT_FILE" "$STDERR_FILE"; then
  exit_code=2
fi

write_summary
trap - EXIT

cat "$SUMMARY_FILE"
exit "$exit_code"
