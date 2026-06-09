#!/usr/bin/env bash
# Run Granite BF16 / plain INT2 / OSCAR INT2 across context presets and merge CSVs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/repo_paths.sh
source "${ROOT}/scripts/lib/repo_paths.sh"
REPO_ROOT="${ROOT}"

MODEL="checkpoints/granite-4.0-1b-base"
MODEL="$(resolve_repo_path "${MODEL}")"
ROT_DIR="rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
ROT_DIR="$(resolve_repo_path "${ROT_DIR}")"
RESULTS_ROOT="results/granite_bench_matrix"
RESULTS_ROOT="$(resolve_repo_path "${RESULTS_ROOT}")"
TAG="$(date +%Y%m%dT%H%M%S)"
PRESETS="short,medium,long,16k,32k"
MODES="bf16,int2,oscar-int2"
DRY_RUN=0
SKIP_EXISTING=0

usage() {
  cat <<'EOF'
Usage: scripts/bench_matrix.sh [options]

Runs oscar-kv-bench for each preset in PRESETS with the same mode list.
32k uses --max-total-tokens 38272 for fair KV pool sizing.

Options:
  --model PATH           Checkpoint (default: checkpoints/granite-4.0-1b-base)
  --rot-dir PATH         OSCAR rotations (default: granite GPQA seq30000 dir)
  --results-root PATH    Parent output directory
  --tag TAG              Run subdirectory name
  --presets LIST         Comma list: short,medium,long,16k,32k
  --modes LIST           Comma list: bf16,int2,oscar-int2,...
  --dry-run              Print commands only
  --skip-existing        Skip preset if bench_*.csv already exists
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$(resolve_repo_path "$2")"; shift 2 ;;
    --rot-dir) ROT_DIR="$(resolve_repo_path "$2")"; shift 2 ;;
    --results-root) RESULTS_ROOT="$(resolve_repo_path "$2")"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --presets) PRESETS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

OUT_DIR="${RESULTS_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/matrix_manifest.json"
echo "{" > "${MANIFEST}"
first_entry=1

append_manifest() {
  local preset="$1" csv="$2" md="$3"
  if [[ "${first_entry}" -eq 0 ]]; then echo "," >> "${MANIFEST}"; fi
  first_entry=0
  printf '  "%s": {"csv": "%s", "md": "%s"}' \
    "${preset}" "$(repo_relative_path "${csv}")" "$(repo_relative_path "${md}")" >> "${MANIFEST}"
}

IFS=',' read -r -a preset_list <<< "${PRESETS}"
for preset in "${preset_list[@]}"; do
  preset="$(echo "${preset}" | xargs)"
  [[ -n "${preset}" ]] || continue
  preset_dir="${OUT_DIR}/${preset}"
  mkdir -p "${preset_dir}"

  if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    if compgen -G "${preset_dir}/bench_*.csv" >/dev/null; then
      echo "[bench_matrix] skip ${preset}: existing CSV"
      csv_path="$(ls -1t "${preset_dir}"/bench_*.csv | head -1)"
      md_path="${csv_path%.csv}.md"
      append_manifest "${preset}" "${csv_path}" "${md_path}"
      continue
    fi
  fi

  extra_args=()
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

  echo "[bench_matrix] preset=${preset}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; echo
    continue
  fi

  "${cmd[@]}"
  csv_path="$(ls -1t "${preset_dir}"/bench_*.csv | head -1)"
  md_path="${csv_path%.csv}.md"
  append_manifest "${preset}" "${csv_path}" "${md_path}"
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "" >> "${MANIFEST}"
  echo "}" >> "${MANIFEST}"
  echo "[bench_matrix] manifest -> ${MANIFEST}"
  if [[ -x "${ROOT}/.venv-oscar-kv/bin/oscar-kv-regression-gate" ]]; then
    for preset in "${preset_list[@]}"; do
      preset="$(echo "${preset}" | xargs)"
      csv_glob="${OUT_DIR}/${preset}/bench_*.csv"
      if compgen -G "${csv_glob}" >/dev/null; then
        echo "[bench_matrix] gate ${preset}"
        "${ROOT}/scripts/regression_gate.sh" --scenario balanced ${csv_glob} || true
      fi
    done
  fi
fi
