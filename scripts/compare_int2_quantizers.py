#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Q2_CENTROIDS = np.array([-0.9816, -0.4528, 0.4528, 0.9816], dtype=np.float32)
Q2_THRESHOLDS = (-0.6745, 0.0, 0.6745)
TURBO2_CENTROIDS = np.array([-0.111724, -0.031626, 0.031626, 0.111724], dtype=np.float32)
TURBO2_THRESHOLDS = (-0.086728, 0.0, 0.086728)
TURBO3_CENTROIDS = np.array(
    [-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685],
    dtype=np.float32,
)
TURBO3_THRESHOLDS = (-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259)
DEFAULT_GGUF = ROOT / "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf"


def q2_0_dequant(x: np.ndarray, group_size: int = 32, clip_ratio: float = 0.0) -> np.ndarray:
    original_shape = x.shape
    if original_shape[-1] % 32 or original_shape[-1] % group_size:
        raise ValueError(f"last dim must be divisible by 32, got {original_shape[-1]}")
    groups = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
    group_mean = groups.mean(axis=-1, keepdims=True)
    grouped = (groups - group_mean).reshape(*original_shape[:-1], original_shape[-1] // 32, 32)
    if clip_ratio > 0.0 and clip_ratio < 1.0:
        flat = grouped.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
        idx = min(int(clip_ratio * group_size), group_size - 1)
        thr = np.sort(np.abs(flat), axis=-1)[..., idx:idx + 1]
        grouped = np.clip(flat, -thr, thr).reshape(*original_shape[:-1], original_shape[-1] // 32, 32)
    mean = grouped.mean(axis=-1, keepdims=True)
    centered = grouped - mean
    sigma = np.maximum(np.sqrt((centered * centered).mean(axis=-1, keepdims=True)), 1e-8)
    scaled = centered / sigma
    q = np.zeros_like(scaled, dtype=np.int64)
    q += (scaled >= Q2_THRESHOLDS[0])
    q += (scaled >= Q2_THRESHOLDS[1])
    q += (scaled >= Q2_THRESHOLDS[2])
    deq = mean + sigma * Q2_CENTROIDS[q]
    deq = deq.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size) + group_mean
    return deq.reshape(original_shape)


def int2_asym_dequant(x: np.ndarray, group_size: int = 128) -> np.ndarray:
    original_shape = x.shape
    if original_shape[-1] % group_size:
        raise ValueError(f"last dim must be divisible by group_size={group_size}, got {original_shape[-1]}")
    grouped = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
    val_min = grouped.min(axis=-1, keepdims=True)
    val_max = grouped.max(axis=-1, keepdims=True)
    scale = np.maximum(val_max - val_min, 1e-8) / 3.0
    zero = -val_min / scale
    q = np.clip(np.round(grouped / scale + zero), 0, 3)
    deq = (q - zero) * scale
    return deq.reshape(original_shape)


def turbo_dequant(x: np.ndarray, centroids: np.ndarray, thresholds: tuple[float, ...], group_size: int) -> np.ndarray:
    original_shape = x.shape
    if original_shape[-1] % group_size:
        raise ValueError(f"last dim must be divisible by group_size={group_size}, got {original_shape[-1]}")
    grouped = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
    norm = np.sqrt((grouped * grouped).sum(axis=-1, keepdims=True))
    inv_norm = np.where(norm > 1e-10, 1.0 / norm, 0.0)
    normalized = grouped * inv_norm
    q = np.zeros_like(normalized, dtype=np.int64)
    for threshold in thresholds:
        q += normalized >= threshold
    recon = centroids[q]
    recon_sq = (recon * recon).sum(axis=-1, keepdims=True)
    recon_norm = np.sqrt(recon_sq)
    corrected = np.where(recon_norm > 1e-10, norm / recon_norm, norm)
    return (corrected * recon).reshape(original_shape)


def turbo2_dequant(x: np.ndarray) -> np.ndarray:
    return turbo_dequant(x, TURBO2_CENTROIDS, TURBO2_THRESHOLDS, 128)


def turbo3_dequant(x: np.ndarray) -> np.ndarray:
    return turbo_dequant(x, TURBO3_CENTROIDS, TURBO3_THRESHOLDS, 32)


def nmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    err = (x_hat - x).astype(np.float32)
    denom = max(float((x.astype(np.float32) * x.astype(np.float32)).mean()), 1e-12)
    return float((err * err).mean() / denom)


def load_gguf_rotations(path: Path, kind: str, n_layer: int) -> dict[int, np.ndarray]:
    sys.path.insert(0, str(ROOT / "third_party/OSCAR/gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    rotations = {}
    for layer in range(n_layer):
        name = f"blk.{layer}.attn_{kind}_rot.weight"
        if name not in tensors:
            raise KeyError(f"missing GGUF rotation tensor: {name}")
        # GGUF stores M.T so ggml_mul_mat(rot, X) computes X @ M.
        rotations[layer] = np.asarray(tensors[name].data, dtype=np.float32).T.copy()
    return rotations


def load_tensor(run_dir: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(run_dir / item["bin"], dtype=np.float32)
    ne = item["ne"]
    expected = int(np.prod(ne))
    if raw.size != expected:
        raise ValueError(f"{item['bin']} has {raw.size} floats, expected {expected}")
    arr = raw.reshape((ne[3], ne[2], ne[1], ne[0]))
    return arr.reshape(-1, ne[0]).astype(np.float32, copy=True)


def summarize(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    vals = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "max": float(vals.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare llama.cpp q2_0 and asymmetric int2 on dumped Granite KV tensors.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/granite_fp16_qkv_calib_20260604T172700Z")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/int2_quantizer_comparison_current")
    args = parser.parse_args()

    manifest = json.loads((args.dump_dir / "manifest.json").read_text())
    items = manifest["items"]
    allowed_prompts = {f"prompt_{i}" for i in range(args.max_prompts)}
    k_rotations = load_gguf_rotations(args.gguf, "k", args.max_layers)
    v_rotations = load_gguf_rotations(args.gguf, "v", args.max_layers)

    rows: list[dict[str, object]] = []
    for item in items:
        if item["kind"] not in {"Kcur", "Vcur"}:
            continue
        if item["prompt"] not in allowed_prompts:
            continue
        layer = int(item["name"].split("-", 1)[1])
        if layer >= args.max_layers:
            continue
        x = load_tensor(args.dump_dir, item)
        rotation = k_rotations[layer] if item["kind"] == "Kcur" else v_rotations[layer]

        x_q2 = q2_0_dequant(x)
        x_asym = int2_asym_dequant(x)
        x_rot = x @ rotation
        x_rot_q2 = q2_0_dequant(x_rot) @ rotation.T
        x_rot_asym = int2_asym_dequant(x_rot) @ rotation.T
        x_turbo2 = turbo2_dequant(x)
        x_turbo3 = turbo3_dequant(x)
        x_rot_turbo2 = turbo2_dequant(x_rot) @ rotation.T
        x_rot_turbo3 = turbo3_dequant(x_rot) @ rotation.T

        rows.append({
            "prompt": item["prompt"],
            "kind": item["kind"],
            "layer": layer,
            "num_rows": x.shape[0],
            "q2_nmse": nmse(x, x_q2),
            "q2_rot_nmse": nmse(x, x_rot_q2),
            "asym_nmse": nmse(x, x_asym),
            "asym_rot_nmse": nmse(x, x_rot_asym),
            "turbo2_nmse": nmse(x, x_turbo2),
            "turbo2_rot_nmse": nmse(x, x_rot_turbo2),
            "turbo3_nmse": nmse(x, x_turbo3),
            "turbo3_rot_nmse": nmse(x, x_rot_turbo3),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "quantizer_nmse.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for kind in ("Kcur", "Vcur"):
        subset = [row for row in rows if row["kind"] == kind]
        out = {"kind": kind, "rows": len(subset)}
        for key in ("q2_nmse", "q2_rot_nmse", "asym_nmse", "asym_rot_nmse", "turbo2_nmse", "turbo2_rot_nmse", "turbo3_nmse", "turbo3_rot_nmse"):
            stats = summarize(subset, key)
            out[f"{key}_mean"] = stats["mean"]
            out[f"{key}_median"] = stats["median"]
            out[f"{key}_max"] = stats["max"]
        out["q2_rot_vs_plain_mean_ratio"] = out["q2_rot_nmse_mean"] / max(out["q2_nmse_mean"], 1e-12)
        out["asym_rot_vs_plain_mean_ratio"] = out["asym_rot_nmse_mean"] / max(out["asym_nmse_mean"], 1e-12)
        out["q2_vs_asym_rot_mean_ratio"] = out["q2_rot_nmse_mean"] / max(out["asym_rot_nmse_mean"], 1e-12)
        out["turbo2_rot_vs_plain_mean_ratio"] = out["turbo2_rot_nmse_mean"] / max(out["turbo2_nmse_mean"], 1e-12)
        out["turbo3_rot_vs_plain_mean_ratio"] = out["turbo3_rot_nmse_mean"] / max(out["turbo3_nmse_mean"], 1e-12)
        out["turbo3rot_vs_q2rot_mean_ratio"] = out["turbo3_rot_nmse_mean"] / max(out["q2_rot_nmse_mean"], 1e-12)
        summary_rows.append(out)

    summary_csv = args.out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md = ["| kind | rows | q2+rot mean | asym+rot mean | turbo2+rot mean | turbo3+rot mean | turbo3rot/q2rot |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        md.append(
            f"| {row['kind']} | {row['rows']} | "
            f"{row['q2_rot_nmse_mean']:.6f} | {row['asym_rot_nmse_mean']:.6f} | "
            f"{row['turbo2_rot_nmse_mean']:.6f} | {row['turbo3_rot_nmse_mean']:.6f} | "
            f"{row['turbo3rot_vs_q2rot_mean_ratio']:.3f} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    (args.out_dir / "README.md").write_text(
        "Offline Granite KV quantizer comparison on activation dump. "
        "q2 is llama.cpp block_q2_0 Lloyd-Max; asym is group-128 asymmetric scale/zero int2; "
        "turbo2/turbo3 are dedicated TurboQuant-style KV block formats.\n"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
