#!/usr/bin/env bash
# Download Granite 4.0 1B Base and/or Gemma 4 E2B into ./checkpoints (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/checkpoints"
HF_REVISION_ARG=""
VENV_DIR="${ROOT}/.venv-oscar-kv"

usage() {
  cat <<'EOF'
Usage: scripts/download_models.sh [options] [all|granite|gemma4]
       scripts/download_models.sh [options] --model all|granite|gemma4

Default: all

Options:
  --checkpoint-root PATH   Output directory (default: checkpoints)
  --revision REV           Hugging Face revision passed to downloads
  --venv-dir PATH          Virtualenv for weight normalization (default: .venv-oscar-kv)
  -h, --help               Show this help

Models:
  granite   ibm-granite/granite-4.0-1b-base
  gemma4    google/gemma-4-E2B
  all       both models
EOF
}

MODEL_SET="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root)
      if [[ $# -lt 2 ]]; then
        echo "--checkpoint-root requires a path" >&2
        usage >&2
        exit 2
      fi
      OUT="$2"
      shift 2
      ;;
    --revision)
      if [[ $# -lt 2 ]]; then
        echo "--revision requires a value" >&2
        usage >&2
        exit 2
      fi
      HF_REVISION_ARG="$2"
      shift 2
      ;;
    --venv-dir)
      if [[ $# -lt 2 ]]; then
        echo "--venv-dir requires a path" >&2
        usage >&2
        exit 2
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "--model requires a value: all, granite, or gemma4" >&2
        usage >&2
        exit 2
      fi
      MODEL_SET="$2"
      shift 2
      ;;
    --model=*)
      MODEL_SET="${1#--model=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    all|granite|granite4|granite-4.0-1b-base|gemma|gemma4|gemma-4-E2B)
      MODEL_SET="$1"
      shift
      ;;
    *)
      echo "Unknown option or model: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODEL_SET" in
  all) ;;
  granite|granite4|granite-4.0-1b-base) MODEL_SET="granite" ;;
  gemma|gemma4|gemma-4-E2B) MODEL_SET="gemma4" ;;
  *)
    echo "Unknown model: ${MODEL_SET}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${OUT}" != /* ]]; then
  OUT="${ROOT}/${OUT}"
fi
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install Hugging Face CLI: pip install 'huggingface_hub[cli]'"
  exit 1
fi

mkdir -p "$OUT"
echo "Downloading to: $OUT"

# HF_TOKEN may still be needed by huggingface-cli for gated models after license acceptance.
common_args=()
if [[ -n "${HF_REVISION_ARG}" ]]; then
  common_args+=(--revision "$HF_REVISION_ARG")
fi

download_model() {
  local repo_id="$1"
  local target_dir="$2"

  normalize_weights() {
    local index_file="${target_dir}/model.safetensors.index.json"
    local consolidated_file="${target_dir}/model.safetensors"
    if [[ ! -f "${index_file}" || ! -f "${consolidated_file}" ]]; then
      return 0
    fi
    "${VENV_DIR}/bin/python" - "${target_dir}" <<'PY' || true
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
index_file = target / "model.safetensors.index.json"
consolidated_file = target / "model.safetensors"

try:
    weight_map = json.loads(index_file.read_text())["weight_map"]
except Exception as exc:
    print(f"Could not read {index_file}: {exc}", file=sys.stderr)
    raise SystemExit(0)

referenced_files = sorted(set(weight_map.values()))
missing_refs = [name for name in referenced_files if not (target / name).is_file()]
if not missing_refs:
    raise SystemExit(0)

any_shard_present = any((target / name).is_file() for name in referenced_files)
if any_shard_present:
    print(
        f"In {target}: index references missing shards {missing_refs!r} "
        "but other shard files exist; re-download the model or fix the layout.",
        file=sys.stderr,
    )
    raise SystemExit(0)

# All shard filenames from the index are absent (common: only model.safetensors was kept).
try:
    from safetensors import safe_open

    with safe_open(consolidated_file, framework="pt", device="cpu") as f:
        consolidated_keys = set(f.keys())
except Exception as exc:
    print(f"Could not inspect {consolidated_file}: {exc}", file=sys.stderr)
    raise SystemExit(0)

needed_keys = set(weight_map.keys())
if not needed_keys <= consolidated_keys:
    print(
        f"{consolidated_file.name} does not contain all tensors listed in the index "
        f"({len(needed_keys - consolidated_keys)} missing); leaving index unchanged.",
        file=sys.stderr,
    )
    raise SystemExit(0)

backup = index_file.with_suffix(index_file.suffix + ".stale")
counter = 1
while backup.exists():
    backup = index_file.with_suffix(index_file.suffix + f".stale.{counter}")
    counter += 1
index_file.rename(backup)
print(
    f"Renamed stale {index_file.name} -> {backup.name}; "
    f"shard files from index were absent; using consolidated {consolidated_file.name}."
)
PY
  }

  has_weights() {
    compgen -G "${target_dir}/*.safetensors" >/dev/null ||
      compgen -G "${target_dir}/*.bin" >/dev/null ||
      compgen -G "${target_dir}/*.pt" >/dev/null
  }

  normalize_weights
  if [[ -f "${target_dir}/config.json" ]] && has_weights; then
    echo "Skipping ${repo_id}; found config and model weights in ${target_dir}"
    return 0
  fi
  if [[ -f "${target_dir}/config.json" ]]; then
    echo "Found ${target_dir}/config.json but no model weights; resuming ${repo_id}"
  fi
  echo "Downloading ${repo_id} -> ${target_dir}"
  huggingface-cli download "${repo_id}" \
    --local-dir "${target_dir}" \
    "${common_args[@]}"

  if [[ ! -f "${target_dir}/config.json" ]]; then
    echo "Download finished but ${target_dir}/config.json is missing" >&2
    exit 1
  fi
  if ! has_weights; then
    echo "Download finished but no model weight file was found in ${target_dir}" >&2
    echo "Expected at least one *.safetensors, *.bin, or *.pt file." >&2
    exit 1
  fi
  normalize_weights
}

case "$MODEL_SET" in
  all)
    download_model ibm-granite/granite-4.0-1b-base "$OUT/granite-4.0-1b-base"
    download_model google/gemma-4-E2B "$OUT/gemma-4-E2B"
    ;;
  granite)
    download_model ibm-granite/granite-4.0-1b-base "$OUT/granite-4.0-1b-base"
    ;;
  gemma4)
    download_model google/gemma-4-E2B "$OUT/gemma-4-E2B"
    ;;
esac

echo "Done. Point --model-path to:"
if [[ "$MODEL_SET" == "all" || "$MODEL_SET" == "granite" ]]; then
  echo "  $OUT/granite-4.0-1b-base"
fi
if [[ "$MODEL_SET" == "all" || "$MODEL_SET" == "gemma4" ]]; then
  echo "  $OUT/gemma-4-E2B"
fi
