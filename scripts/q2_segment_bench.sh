#!/usr/bin/env bash
# Segment q2/q2 FA cost via cache-type isolation on the supported q2/q2 vec path.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${LLAMA_BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
MODEL="${MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_DIR:-$ROOT_DIR/runs/q2_segments_${STAMP}}"
DRY_RUN="${DRY_RUN:-1}"
mkdir -p "$OUT"

bench_one() {
  local tag=$1 k=$2 v=$3 p=$4 timeout_sec=${5:-300}
  echo "=== $tag p=$p K=$k V=$v ===" | tee "$OUT/${tag}.log"
  if timeout "$timeout_sec" "$BENCH" -m "$MODEL" -p "$p" -n 64 -r 1 -ngl 999 -fa 1 \
      --cache-type-k "$k" --cache-type-v "$v" 2>&1 | tee -a "$OUT/${tag}.log"; then
    echo "status=ok" >> "$OUT/${tag}.log"
  else
    echo "status=timeout_or_fail exit=$?" >> "$OUT/${tag}.log"
  fi
}

{
  echo "model=$MODEL"
  echo "bench=$BENCH"
  echo "note=q2/q2 uses vec FA; mixed K/V types use slow unsupported paths and are not comparable"
} > "$OUT/config.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  {
    echo "dry_run=1"
    echo "out=$OUT"
    echo "bench=$BENCH"
    echo "model=$MODEL"
    echo "Set DRY_RUN=0 to run segment microbench."
  } | tee "$OUT/dry_run.txt"
  echo "OUT=$OUT"
  exit 0
fi

# Primary baseline (supported fast path)
bench_one q2q2_pp8192 q2_0 q2_0 8192 600
bench_one q4q4_pp8192 q4_0 q4_0 8192 600
bench_one q2q2_pp2048 q2_0 q2_0 2048 300
bench_one q2q2_pp512  q2_0 q2_0 512  120

# Decode vs prefill ratio from same run (tg64 vs pp8192 in q2q2_pp8192)

# Mixed paths: document as unsupported/slow (short timeout, smaller prompt)
bench_one q2f16_pp512  q2_0 f16  512  120
bench_one f16q2_pp512  f16  q2_0 512  120

python3 - "$OUT" <<'PY'
import glob, re, json, sys
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"{out}/*.log")):
    text = open(path).read()
    tag = path.rsplit("/", 1)[-1].replace(".log", "")
    status = "ok" if "status=ok" in text else "fail"
    for m in re.finditer(r"pp(\d+)\s*\|\s*([0-9.]+)", text):
        rows.append({"tag": tag, "metric": f"pp{m.group(1)}", "tps": float(m.group(2)), "status": status})
    for m in re.finditer(r"tg(\d+)\s*\|\s*([0-9.]+)", text):
        rows.append({"tag": tag, "metric": f"tg{m.group(1)}", "tps": float(m.group(2)), "status": status})
summary = {"out": out, "rows": rows}
open(f"{out}/summary.json", "w").write(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "OUT=$OUT"
