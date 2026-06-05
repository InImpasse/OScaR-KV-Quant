#!/usr/bin/env bash
# Download Granite 4.0 1B Base and Gemma 4 E2B into ./checkpoints (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"

mkdir -p "$OUT"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install Hugging Face CLI: pip install 'huggingface_hub[cli]'"
  exit 1
fi

echo "Downloading to: $OUT"

# Optional: HF_TOKEN in environment for gated models (Gemma may require license acceptance on HF).
common_args=()
if [[ -n "${HF_REVISION:-}" ]]; then
  common_args+=(--revision "$HF_REVISION")
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
    "${VIRTUAL_ENV:-${ROOT}/.venv-oscar-kv}/bin/python" - "${target_dir}" <<'PY' || true
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

try:
    from safetensors import safe_open

    with safe_open(consolidated_file, framework="pt", device="cpu") as f:
        consolidated_keys = set(f.keys())
except Exception as exc:
    print(f"Could not inspect {consolidated_file}: {exc}", file=sys.stderr)
    raise SystemExit(0)

if consolidated_keys != set(weight_map):
    print(
        f"Found {consolidated_file.name} and stale index refs, but key sets differ; "
        "leaving files unchanged.",
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
    f"using consolidated {consolidated_file.name}."
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

download_model ibm-granite/granite-4.0-1b-base "$OUT/granite-4.0-1b-base"
download_model google/gemma-4-E2B "$OUT/gemma-4-E2B"

echo "Done. Point OSCAR_KV_MODEL_PATH or --model-path to:"
echo "  $OUT/granite-4.0-1b-base"
echo "  $OUT/gemma-4-E2B"
