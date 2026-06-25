#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT_DIR/third_party/OSCAR/build-cuda/bin/llama-bench}"
MODEL="${MODEL:-$ROOT_DIR/checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf}"
PROMPTS="${PROMPTS:-512,2048,8192}"
RUN_REAL="${RUN_REAL:-0}"

if [[ "$RUN_REAL" != "1" ]]; then
  echo "Dry run: set RUN_REAL=1 to execute."
fi

IFS=',' read -ra PROMPT_ARR <<< "$PROMPTS"
for prompt in "${PROMPT_ARR[@]}"; do
  for mode in vec tile mixed_vec; do
    case "$mode" in
      vec) env=(LLAMA_KV_Q2_TILE_MAIN=0 LLAMA_KV_Q2_TILE_MIXED=0) ;;
      tile) env=(LLAMA_KV_Q2_TILE_MAIN=1 LLAMA_KV_Q2_TILE_MIXED=0) ;;
      mixed_vec) env=(LLAMA_KV_Q2_TILE_MAIN=0 LLAMA_KV_Q2_TILE_MIXED=1 LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1 LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_Q2_0_OWHT=1) ;;
    esac
    echo "# $mode pp$prompt"
    if [[ "$RUN_REAL" == "1" ]]; then
      env "${env[@]}" "$BENCH" -m "$MODEL" -p "$prompt" -n 1 -r 1 -ngl 999 -fa 1 \
        --cache-type-k q2_0 --cache-type-v q2_0 --output json \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(round([x['avg_ts'] for x in d if x.get('n_prompt',0)>0][0],1))"
    else
      printf ' %q' env "${env[@]}" "$BENCH" -m "$MODEL" -p "$prompt" -n 1 -r 1 -ngl 999 -fa 1 --cache-type-k q2_0 --cache-type-v q2_0
      printf '\n'
    fi
  done
done
