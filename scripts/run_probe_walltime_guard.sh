#!/usr/bin/env bash
# Kill conflicting benches, run probe_walltime_binary.py, resume until 6 jobs done.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBDIR="${1:-probe_walltime_binary_granite_agent}"
RESULTS="${ROOT}/results/${SUBDIR}"
LOG="${RESULTS}/guard.log"
mkdir -p "${RESULTS}"

kill_conflicts() {
  # Avoid broad patterns like "probe_walltime" — they can match this shell's argv.
  pkill -9 -f "\.venv-oscar-kv/bin/oscar-kv-bench" 2>/dev/null || true
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang::scheduler" 2>/dev/null || true
  pkill -9 -f "sglang::detokenizer" 2>/dev/null || true
  sleep 2
}

count_done() {
  if [[ -f "${RESULTS}/summary.jsonl" ]]; then
    wc -l < "${RESULTS}/summary.jsonl"
  else
    echo 0
  fi
}

log() {
  echo "$(date -Iseconds) $*" | tee -a "${LOG}"
}

log "guard start subdir=${SUBDIR}"
while true; do
  done_n=$(count_done)
  if [[ "${done_n}" -ge 6 ]]; then
    log "all 6 jobs complete"
    exit 0
  fi
  kill_conflicts
  log "resume probe done=${done_n}/6"
  set +e
  PYTHONUNBUFFERED=1 python3 -u "${ROOT}/scripts/probe_walltime_binary.py" \
    --results-subdir "${SUBDIR}" 2>&1 | tee -a "${RESULTS}/nohup.log"
  rc=${PIPESTATUS[0]}
  set -e
  done_n=$(count_done)
  log "probe exit rc=${rc} done=${done_n}/6"
  if [[ "${done_n}" -ge 6 ]]; then
    log "finished"
    exit 0
  fi
  log "restarting in 10s"
  sleep 10
done
