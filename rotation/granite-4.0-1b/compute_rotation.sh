#!/usr/bin/env bash
# Fit Granite 4.0 1B OSCAR K/V rotations from calibrated Q/K/V dumps.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib/repo_paths.sh
source "${REPO_ROOT}/scripts/lib/repo_paths.sh"
OSCAR_ROOT="${REPO_ROOT}/third_party/OSCAR"
COMPUTE_SCRIPT="${OSCAR_ROOT}/rotation/compute_kv_rotation.py"
MODEL="${MODEL:-checkpoints/granite-4.0-1b-base}"
MODEL="$(resolve_repo_path "${MODEL}")"

METHOD="${METHOD:-qqt_sst}"
read_config_value() {
  local key="$1"
  "${PY:-python3}" - "${MODEL}/config.json" "${key}" <<'PYEOF'
import json, sys
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
if key == "head_dim":
    value = cfg.get("head_dim")
    if value is None and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        value = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
elif key == "num_attention_layers":
    layer_types = cfg.get("layer_types")
    value = sum(1 for item in layer_types if "attention" in str(item)) if isinstance(layer_types, list) else cfg.get("num_hidden_layers")
else:
    value = cfg.get(key)
if value is None:
    raise SystemExit(1)
print(value)
PYEOF
}

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[compute_rotation granite] missing model config: ${MODEL}/config.json" >&2
  echo "  Set MODEL=/path/to/granite checkpoint or run scripts/download_models.sh" >&2
  exit 1
fi

HEAD_DIM="${HEAD_DIM:-$(read_config_value head_dim)}"
NUM_LAYERS="${NUM_LAYERS:-$(read_config_value num_attention_layers)}"
COMPOSITION="${COMPOSITION:-r_h_pbr}"
CHUNK_ID="${CHUNK_ID:-all}"
DATASET="${DATASET:-GPQA}"
CALIB_PROFILE="paper"
ALLOW_WEAK_CALIBRATION=0

usage() {
  cat <<'EOF'
Usage: rotation/granite-4.0-1b/compute_rotation.sh [options]

Options:
  --calib-profile PROFILE     paper | smoke (default: paper)
  --allow-weak-calibration    Allow smoke/debug rotations with weak calibration
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --calib-profile) CALIB_PROFILE="$2"; shift 2 ;;
    --allow-weak-calibration) ALLOW_WEAK_CALIBRATION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${CALIB_PROFILE}" == "paper" ]]; then
  DEFAULT_MIN_CALIB_TOKENS=27000
else
  DEFAULT_MIN_CALIB_TOKENS=1
fi
MIN_CALIB_TOKENS="${MIN_CALIB_TOKENS:-${DEFAULT_MIN_CALIB_TOKENS}}"
MIN_CALIB_PROMPTS="${MIN_CALIB_PROMPTS:-1}"
if [[ -z "${CALIB_DIR:-}" ]]; then
  CALIB_DIR="$(ls -1dt "${SCRIPT_DIR}/${DATASET}"/seq*_prompt*_group*/ 2>/dev/null | head -1 | sed 's:/$::')"
fi
if [[ -z "${CALIB_DIR:-}" ]]; then
  echo "[compute_rotation granite] no calibration dump found under ${SCRIPT_DIR}/${DATASET}" >&2
  echo "  Run save_qkv_granite.sh first, or set CALIB_DIR=/path/to/seq*_prompt*_group*" >&2
  exit 1
fi
DUMP_PATH="${DUMP_PATH:-${CALIB_DIR}/qkv_dumps/gpqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIB_DIR}/rotations}"
export DUMP_PATH
echo "[compute_rotation granite] calib_dir=${CALIB_DIR}"
echo "[compute_rotation granite] dump_path=${DUMP_PATH}"
echo "[compute_rotation granite] output_dir=${OUTPUT_DIR}"
echo "[compute_rotation granite] method=${METHOD} composition=${COMPOSITION} chunk=${CHUNK_ID}"
echo "[compute_rotation granite] head_dim=${HEAD_DIM} num_layers=${NUM_LAYERS}"
echo "[compute_rotation granite] note: GraniteMoeHybrid may have fewer attention layers than total hidden layers; verify layer_* dump dirs if rotation fails."

PY="${PY:-python3}"
if [[ "${METHOD}" != "hadamard" && ! -d "${DUMP_PATH}" ]]; then
  echo "[compute_rotation granite] dump path does not exist: ${DUMP_PATH}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

if [[ "${METHOD}" != "hadamard" ]]; then
  "${PY}" - "${CALIB_DIR}" "${DUMP_PATH}" "${NUM_LAYERS}" "${MIN_CALIB_TOKENS}" "${MIN_CALIB_PROMPTS}" "${ALLOW_WEAK_CALIBRATION}" <<'PYEOF'
import json
import sys
from pathlib import Path

import torch

calib_dir = Path(sys.argv[1])
dump_path = Path(sys.argv[2])
expected_layers = int(sys.argv[3])
min_tokens = int(sys.argv[4])
min_prompts = int(sys.argv[5])
allow_weak = sys.argv[6] == "1"

layers = sorted(
    [p for p in dump_path.iterdir() if p.is_dir() and p.name.startswith("layer_")],
    key=lambda p: int(p.name.split("_", 1)[1]),
)
if len(layers) != expected_layers:
    raise SystemExit(
        f"expected {expected_layers} layer dumps, found {len(layers)} under {dump_path}"
    )
for layer in layers:
    for name in ("q", "k", "v", "seq_lens"):
        if not (layer / name).is_dir():
            raise SystemExit(f"missing {layer / name}")

seq_dir = dump_path / "layer_0" / "seq_lens"
prompt_count = 0
token_count = 0
for path in sorted(seq_dir.glob("*.pt"), key=lambda p: int(p.stem)):
    seq = torch.load(path, weights_only=True, map_location="cpu")
    prompt_count += len(seq.tolist())
    token_count += int(seq.sum().item())

meta_path = calib_dir / "calibration_meta.json"
meta = {}
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    token_count = int(meta.get("dumped_tokens", token_count))
    prompt_count = int(meta.get("num_prompts_captured", prompt_count))

weak = token_count < min_tokens or prompt_count < min_prompts
if weak and not allow_weak:
    raise SystemExit(
        "weak calibration: "
        f"tokens={token_count} prompts={prompt_count}; "
        f"require tokens>={min_tokens} prompts>={min_prompts}. "
        "Use --allow-weak-calibration only for smoke/debug rotations."
    )
print(
    f"[compute_rotation granite] calib_check layers={len(layers)} "
    f"tokens={token_count} prompts={prompt_count}"
)
PYEOF
fi

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

"${PY}" - "${OUTPUT_DIR}" "${METHOD}" "${COMPOSITION}" "${HEAD_DIM}" "${NUM_LAYERS}" "${CALIB_DIR}" "${DUMP_PATH}" "${REPO_ROOT}" <<'PYEOF'
import json
import sys
import time
from pathlib import Path

import torch

output_dir = Path(sys.argv[1])
method = sys.argv[2]
composition = sys.argv[3]
head_dim = int(sys.argv[4])
num_layers = int(sys.argv[5])
calib_dir = Path(sys.argv[6])
dump_path = Path(sys.argv[7])
repo_root = Path(sys.argv[8]).resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()

if method == "hadamard":
    files = {
        "k": output_dir / "k_rotation_hadamard.pt",
        "v": output_dir / "v_rotation_hadamard.pt",
    }
else:
    files = {
        "k": output_dir / f"k_rotation_{method.split('_')[0]}_{composition}.pt",
        "v": output_dir / f"v_rotation_{method.split('_')[-1]}_{composition}.pt",
    }

stats = {}
eye = torch.eye(head_dim)
for target, path in files.items():
    if not path.exists():
        raise SystemExit(f"missing expected rotation file: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    layers = state.get("layers", {})
    if len(layers) != num_layers:
        raise SystemExit(f"{path} has {len(layers)} layers, expected {num_layers}")
    max_err = 0.0
    for layer_id in range(num_layers):
        entry = layers.get(layer_id, layers.get(str(layer_id)))
        if entry is None:
            raise SystemExit(f"{path} missing layer {layer_id}")
        rot = entry["rotation"].float()
        if tuple(rot.shape) != (head_dim, head_dim):
            raise SystemExit(f"{path} layer {layer_id} shape {tuple(rot.shape)}")
        err = (rot @ rot.T - eye).abs().max().item()
        max_err = max(max_err, err)
    stats[target] = {"path": rel(path), "max_orthogonality_error": max_err}

calib_meta_path = calib_dir / "calibration_meta.json"
calib_meta = json.loads(calib_meta_path.read_text()) if calib_meta_path.exists() else {}
if calib_meta.get("model"):
    model_path = Path(calib_meta["model"])
    if model_path.is_absolute():
        calib_meta = {**calib_meta, "model": rel(model_path)}
model_ref = calib_meta.get("model")
if model_ref:
    model_path = Path(model_ref)
    model_ref = rel(model_path) if model_path.is_absolute() else str(model_ref)
meta = {
    "format_version": 1,
    "model": model_ref,
    "dataset": calib_meta.get("dataset", "GPQA"),
    "method": method,
    "composition": composition,
    "recipe": "OSCAR qqt_sst calibrated spectral covariance + Hadamard + bit-reversal",
    "head_dim": head_dim,
    "num_layers": num_layers,
    "dump_path": rel(dump_path),
    "calibration": calib_meta,
    "rotation_files": stats,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
(output_dir / "rotation_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"[compute_rotation granite] wrote {output_dir / 'rotation_meta.json'}")
for target, stat in stats.items():
    print(
        f"[compute_rotation granite] {target} max_orthogonality_error="
        f"{stat['max_orthogonality_error']:.3e}"
    )
PYEOF

ls -la "${OUTPUT_DIR}" | grep -E "rotation.*\.(pt|json)" || true
