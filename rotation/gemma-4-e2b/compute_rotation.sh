#!/usr/bin/env bash
# Fit K/V rotations for Gemma 4 E2B — set NUM_LAYERS / HEAD_DIM from checkpoint config.json if defaults wrong.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"
COMPUTE_SCRIPT="${OSCAR_ROOT}/rotation/compute_kv_rotation.py"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/gemma-4-E2B}"
PY="${PY:-python3}"

METHOD="${METHOD:-qqt_sst}"
read_config_value() {
  local key="$1"
  "${PY}" - "${MODEL}/config.json" "${key}" <<'PYEOF'
import json, sys
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
if key == "head_dim":
    value = text_cfg.get("head_dim")
    if value is None and text_cfg.get("hidden_size") and text_cfg.get("num_attention_heads"):
        value = int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"])
elif key == "num_attention_layers":
    layer_types = text_cfg.get("layer_types")
    value = sum(1 for item in layer_types if "attention" in str(item)) if isinstance(layer_types, list) else text_cfg.get("num_hidden_layers")
else:
    value = text_cfg.get(key)
if value is None:
    raise SystemExit(1)
print(value)
PYEOF
}

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[compute_rotation gemma4] missing model config: ${MODEL}/config.json" >&2
  echo "  Set MODEL=/path/to/gemma checkpoint or run scripts/download_models.sh" >&2
  exit 1
fi

HEAD_DIM="${HEAD_DIM:-$(read_config_value head_dim)}"
NUM_LAYERS="${NUM_LAYERS:-$(read_config_value num_attention_layers)}"
COMPOSITION="${COMPOSITION:-r_h_pbr}"
CHUNK_ID="${CHUNK_ID:-all}"
DATASET="${DATASET:-GPQA}"
if [[ -z "${CALIB_DIR:-}" ]]; then
  CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
if [[ -z "${CALIB_DIR:-}" ]]; then
  echo "[compute_rotation gemma4] no calibration dump found under ${SCRIPT_DIR}/${DATASET}" >&2
  echo "  Run save_qkv_gemma4.sh first, or set CALIB_DIR=/path/to/seq*_prompt*_group*" >&2
  exit 1
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
export DUMP_PATH
echo "[compute_rotation gemma4] calib_dir=${CALIB_DIR} dump_path=${DUMP_PATH}"
echo "[compute_rotation gemma4] head_dim=${HEAD_DIM} num_layers=${NUM_LAYERS}"

if [[ "${METHOD}" != "hadamard" && ! -d "${DUMP_PATH}" ]]; then
  echo "[compute_rotation gemma4] dump path does not exist: ${DUMP_PATH}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

case "${METHOD}" in
  hadamard)
    "${PY}" "${COMPUTE_SCRIPT}" \
      --method hadamard \
      --head-dim "${HEAD_DIM}" \
      --num-layers "${NUM_LAYERS}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  qqt_sst|ktk_vtv|qqt|sst|ktk|vtv|uresidual)
    "${PY}" "${COMPUTE_SCRIPT}" \
      --dump-path "${DUMP_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --head-dim "${HEAD_DIM}" \
      --chunk-id "${CHUNK_ID}" \
      --method "${METHOD}" \
      --composition "${COMPOSITION}"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 1
    ;;
esac
ls -la "${OUTPUT_DIR}" | grep -E "rotation.*\.pt" || true
