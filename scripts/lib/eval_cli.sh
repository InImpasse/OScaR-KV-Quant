#!/usr/bin/env bash
# Shared --flag parsing for rotation/granite-4.0-1b eval wrappers.
# Scripts set defaults, then call parse_eval_cli_args "$@" and read the variables.

eval_cli_usage_gpqa() {
  cat <<'EOF'
Usage: eval_gpqa_granite_<mode>.sh [options]

Options:
  --num-examples N     Number of GPQA-Diamond examples (default: 198)
  --model PATH         Model checkpoint directory
  --rot-dir PATH       Rotation directory (OSCAR INT2 only)
  --run-dir PATH       Output directory for this run
  --port N             SGLang HTTP port
  --gpu N              CUDA device index
  --max-new-tokens N   Generation length per example
  -h, --help           Show this help
EOF
}

eval_cli_usage_gsm8k() {
  cat <<'EOF'
Usage: eval_gsm8k_granite.sh [options]

Options:
  --mode MODE          bf16 | int2 | oscar-int2 (default: bf16)
  --num-questions N    GSM8K test questions (default: 200)
  --model PATH         Model checkpoint directory
  --rot-dir PATH       Rotation directory (OSCAR INT2 only)
  --run-dir PATH       Output directory
  --port N             SGLang HTTP port
  --gpu N              CUDA device index
  --max-running-requests N
                       SGLang concurrent request capacity (default: 1)
  --data-path PATH     GSM8K JSONL path
  -h, --help           Show this help
EOF
}

eval_cli_usage_simple() {
  cat <<'EOF'
Usage: eval_simple_suite_granite.sh [options]

Options:
  --mode MODE          bf16 | int2 | oscar-int2 (default: bf16)
  --task TASK          humaneval | math | aime25 | gpqa | gsm8k
  --num-examples N     Subset size (task default if omitted)
  --repeat N           Repeat count per example
  --num-threads N      Eval parallelism
  --max-running-requests N   Forwarded to SGLang --max-running-requests (default: 1 or env)
  --max-queued-requests N    Forwarded to SGLang --max-queued-requests (default: 4 or env)
  --model PATH         Model checkpoint directory
  --rot-dir PATH       Rotation directory (OSCAR INT2 only)
  --run-dir PATH       Output directory
  --port N             SGLang HTTP port
  --gpu N              CUDA device index
  --max-new-tokens N   Generation length
  --post-ready-sleep N Delay after server readiness
  -h, --help           Show this help
EOF
}

eval_cli_usage_suite() {
  cat <<'EOF'
Usage: eval_accuracy_suite.sh [options]

Runs BF16 / plain INT2 / OSCAR INT2 on the selected Granite benchmarks and
writes comparison JSON under rotation/granite-4.0-1b/accuracy_suite_<tag>/.

Options:
  --quick                     Local regression: skip GPQA/GSM8K, run HumanEval
                              and MATH-500 with 20 examples each (see README)
  --gpqa-num-examples N       GPQA examples (default: 198, 0 = skip)
  --gsm8k-num-questions N     GSM8K questions (default: 200, 0 = skip)
  --humaneval-num-examples N  HumanEval problems (default: 164, 0 = skip)
  --math-num-examples N       MATH-500 problems (default: 500, 0 = skip)
  --rot-dir PATH              OSCAR rotation directory
  --model PATH                Model checkpoint directory
  --tag TAG                   Suite output tag (default: timestamp)
  -h, --help                  Show this help
EOF
}

parse_eval_cli_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --num-examples)
        NUM_EXAMPLES="$2"
        shift 2
        ;;
      --num-questions)
        NUM_QUESTIONS="$2"
        shift 2
        ;;
      --mode)
        MODE="$2"
        shift 2
        ;;
      --task)
        TASK="$2"
        shift 2
        ;;
      --model)
        MODEL="$2"
        shift 2
        ;;
      --rot-dir)
        ROT_DIR="$2"
        shift 2
        ;;
      --run-dir)
        RUN_DIR="$2"
        shift 2
        ;;
      --port)
        PORT="$2"
        shift 2
        ;;
      --gpu)
        GPU="$2"
        shift 2
        ;;
      --max-new-tokens)
        MAX_NEW_TOKENS="$2"
        shift 2
        ;;
      --post-ready-sleep)
        POST_READY_SLEEP="$2"
        shift 2
        ;;
      --max-running-requests)
        MAX_RUNNING_REQUESTS="$2"
        shift 2
        ;;
      --max-queued-requests)
        MAX_QUEUED_REQUESTS="$2"
        shift 2
        ;;
      --k-rotation-filename)
        K_ROT_FILENAME="$2"
        shift 2
        ;;
      --v-rotation-filename)
        V_ROT_FILENAME="$2"
        shift 2
        ;;
      --mixed-kv-hp-max-splits)
        MIXED_KV_HP_MAX_SPLITS="$2"
        shift 2
        ;;
      --mixed-kv-prefix-tokens)
        MIXED_KV_PREFIX_TOKENS="$2"
        shift 2
        ;;
      --mixed-kv-recent-tokens)
        MIXED_KV_RECENT_TOKENS="$2"
        shift 2
        ;;
      --mixed-kv-max-quant-tokens)
        MIXED_KV_MAX_QUANT_TOKENS="$2"
        shift 2
        ;;
      --mixed-kv-hp-prefix-pool-tokens)
        MIXED_KV_HP_PREFIX_POOL_TOKENS="$2"
        shift 2
        ;;
      --mixed-kv-scale-dtype)
        MIXED_KV_SCALE_DTYPE="$2"
        shift 2
        ;;
      --oscar-k-clip-ratio)
        OSCAR_K_CLIP_RATIO="$2"
        shift 2
        ;;
      --oscar-v-clip-ratio)
        OSCAR_V_CLIP_RATIO="$2"
        shift 2
        ;;
      --lloyd-max)
        LLOYD_MAX="$2"
        shift 2
        ;;
      --enable-fused-rotate-clip-quant)
        OSCAR_FUSED_ROTATE_CLIP_QUANT=1
        shift
        ;;
      --disable-fused-rotate-clip-quant)
        OSCAR_FUSED_ROTATE_CLIP_QUANT=0
        shift
        ;;
      --repeat)
        REPEAT="$2"
        shift 2
        ;;
      --num-threads)
        NUM_THREADS="$2"
        shift 2
        ;;
      --data-path)
        DATA_PATH="$2"
        shift 2
        ;;
      --gpqa-num-examples)
        GPQA_N="$2"
        shift 2
        ;;
      --gsm8k-num-questions)
        GSM8K_N="$2"
        shift 2
        ;;
      --humaneval-num-examples)
        HUMANEVAL_N="$2"
        shift 2
        ;;
      --math-num-examples)
        MATH_N="$2"
        shift 2
        ;;
      --tag)
        SUITE_TAG="$2"
        shift 2
        ;;
      --quick)
        QUICK_SUITE=1
        shift
        ;;
      -h|--help)
        if declare -F "${EVAL_CLI_HELP_FN:-eval_cli_usage_gpqa}" >/dev/null 2>&1; then
          "${EVAL_CLI_HELP_FN:-eval_cli_usage_gpqa}"
        else
          eval_cli_usage_gpqa
        fi
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        echo "unknown option: $1" >&2
        exit 2
        ;;
      *)
        break
        ;;
    esac
  done
  EVAL_CLI_REMAINING=("$@")
}
