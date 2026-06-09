#!/usr/bin/env bash
# Run Granite BF16 / INT2 / OSCAR-INT2 across presets with CUDA graph off vs on.
# Writes results under results/cuda_graph_compare_matrix/<TAG>/{off,on}/<preset>/bench_*.csv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/repo_paths.sh
source "${ROOT}/scripts/lib/repo_paths.sh"
REPO_ROOT="${ROOT}"

MODEL="checkpoints/granite-4.0-1b-base"
MODEL="$(resolve_repo_path "${MODEL}")"
ROT_DIR="rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
MATRIX_ROOT="${ROOT}/results/cuda_graph_compare_matrix"
MATRIX_ROOT="$(resolve_repo_path "${MATRIX_ROOT}")"
TAG="$(date +%Y%m%dT%H%M%S)"
PRESETS="${PRESETS:-short,medium,long,16k,32k}"
MODES="${MODES:-bf16,int2,oscar-int2}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/cuda_graph_compare_matrix.sh [options]

Runs ./scripts/bench.sh for each (CUDA graph off|on) × preset × mode list.
Uses --enable-cuda-graph for the "on" half (SGLang default graph path enabled).
Uses --disable-oscar-cuda-graph for the "off" half so OSCAR is also graph-off.

Options:
  --tag TAG              Subdirectory under results/cuda_graph_compare_matrix/
  --presets LIST         Comma list (default: short,medium,long,16k,32k)
  --modes LIST           Comma list (default: bf16,int2,oscar-int2)
  --model PATH           Checkpoint
  --rot-dir PATH         OSCAR rotations
  --matrix-root PATH     Parent of TAG directory (default: results/cuda_graph_compare_matrix)
  --dry-run              Print commands only
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --presets) PRESETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --model) MODEL="$(resolve_repo_path "$2")"; shift 2 ;;
    --rot-dir) ROT_DIR="$(resolve_repo_path "$2")"; shift 2 ;;
    --matrix-root) MATRIX_ROOT="$(resolve_repo_path "$2")"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

OUT_BASE="${MATRIX_ROOT}/${TAG}"
mkdir -p "${OUT_BASE}"

IFS=',' read -r -a preset_list <<< "${PRESETS}"

for graph_state in off on; do
  cg_args=()
  if [[ "${graph_state}" == "on" ]]; then
    cg_args+=(--enable-cuda-graph)
  else
    cg_args+=(--disable-oscar-cuda-graph)
  fi

  for preset in "${preset_list[@]}"; do
    preset="$(echo "${preset}" | xargs)"
    [[ -n "${preset}" ]] || continue
    preset_dir="${OUT_BASE}/${graph_state}/${preset}"
    mkdir -p "${preset_dir}"

    extra_args=("${cg_args[@]}")
    if [[ "${preset}" == "16k" ]]; then
      extra_args+=(--max-total-tokens 17408)
    fi
    if [[ "${preset}" == "32k" ]]; then
      extra_args+=(--max-total-tokens 38272)
    fi
    if [[ "${MODES}" == *"oscar"* ]]; then
      extra_args+=(--rot-dir "${ROT_DIR}")
    fi

    cmd=(
      "${ROOT}/scripts/bench.sh"
      --profile granite
      --model-path "${MODEL}"
      --preset "${preset}"
      --modes "${MODES}"
      --request-api completions
      --results-dir "${preset_dir}"
      "${extra_args[@]}"
    )

    echo "[cuda_graph_compare_matrix] graph=${graph_state} preset=${preset}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '  '; printf '%q ' "${cmd[@]}"; echo
      continue
    fi
    "${cmd[@]}"
  done
done

echo "[cuda_graph_compare_matrix] done -> ${OUT_BASE}"
