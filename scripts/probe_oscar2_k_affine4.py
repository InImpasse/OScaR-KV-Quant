#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_int2_quantizers import DEFAULT_GGUF, load_gguf_rotations
from compare_int2_attention_error import load_tensor


ROOT = Path(__file__).resolve().parents[1]

OSCAR2_K_CENTROIDS = np.array(
    [-1.35, -0.86, -0.52, -0.185, 0.185, 0.52, 0.86, 1.35],
    dtype=np.float32,
)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    return float(np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-30))


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    err = y.astype(np.float64) - x.astype(np.float64)
    denom = max(float(np.mean(x.astype(np.float64) ** 2)), 1e-30)
    return float(np.mean(err * err) / denom)


def oscar2_k_dequant(x: np.ndarray) -> np.ndarray:
    original_shape = x.shape
    if original_shape[-1] != 128:
        raise ValueError(f"expected D=128, got shape {original_shape}")
    xg = x.reshape(-1, 128).astype(np.float32, copy=False)
    d = np.sqrt(np.mean(xg * xg, axis=-1, keepdims=True))
    inv_d = np.where(d > 1e-8, 1.0 / d, 0.0)
    scaled = xg * inv_d
    idx = np.argmin(np.abs(scaled[..., None] - OSCAR2_K_CENTROIDS), axis=-1)
    out = d * OSCAR2_K_CENTROIDS[idx]
    return out.reshape(original_shape).astype(np.float32, copy=False)


def affine4_reconstruct(k_ref: np.ndarray, k_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if k_ref.shape != k_hat.shape or k_ref.shape[-1] != 128:
        raise ValueError(f"expected matching D=128 tensors, got {k_ref.shape} and {k_hat.shape}")
    ref = k_ref.reshape(-1, 4, 32).astype(np.float32, copy=False)
    hat = k_hat.reshape(-1, 4, 32).astype(np.float32, copy=False)
    mean_hat = hat.mean(axis=-1, keepdims=True)
    mean_ref = ref.mean(axis=-1, keepdims=True)
    var_hat = np.mean((hat - mean_hat) * (hat - mean_hat), axis=-1, keepdims=True)
    cov = np.mean((hat - mean_hat) * (ref - mean_ref), axis=-1, keepdims=True)
    a = np.where(var_hat > 1e-12, cov / var_hat, 1.0)
    b = mean_ref - a * mean_hat
    recon = (a * hat + b).reshape(k_ref.shape)
    coeff = np.concatenate([a.reshape(-1, 4), b.reshape(-1, 4)], axis=-1)
    return recon.astype(np.float32, copy=False), coeff.astype(np.float32, copy=False)


def kq_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    tokens, n_heads, dim = q.shape
    kv_heads = k.shape[1]
    group = n_heads // kv_heads
    out = np.empty((n_heads, tokens, tokens), dtype=np.float32)
    for h in range(n_heads):
        out[h] = q[:, h, :] @ k[:, h // group, :].T
    return out


def layer_items(manifest: dict) -> dict[tuple[str, int, str], dict]:
    by_key = {}
    for item in manifest["items"]:
        if item["kind"] in {"Qcur", "Kcur"}:
            layer = int(item["name"].split("-", 1)[1])
            by_key[(item["prompt"], layer, item["kind"])] = item
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OSCAR2 K affine4 sidecar impact on KQ score error.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/granite_fp16_qkv_calib_20260604T172700Z")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--rotation-mode", choices=("rotated", "plain", "both"), default="both")
    parser.add_argument("--min-improvement", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/oscar2_k_affine4_probe_current")
    args = parser.parse_args()

    manifest = json.loads((args.dump_dir / "manifest.json").read_text())
    by_key = layer_items(manifest)
    rotations = load_gguf_rotations(args.gguf, "k", args.max_layers)
    modes = ("plain", "rotated") if args.rotation_mode == "both" else (args.rotation_mode,)

    rows: list[dict[str, object]] = []
    for prompt_idx in range(args.max_prompts):
        prompt = f"prompt_{prompt_idx}"
        for layer in range(args.max_layers):
            try:
                q = load_tensor(args.dump_dir, by_key[(prompt, layer, "Qcur")])
                k = load_tensor(args.dump_dir, by_key[(prompt, layer, "Kcur")])
            except KeyError:
                continue
            q = q.reshape(-1, q.shape[-2], q.shape[-1]).astype(np.float32, copy=False)
            k = k.reshape(-1, k.shape[-2], k.shape[-1]).astype(np.float32, copy=False)
            for mode in modes:
                if mode == "rotated":
                    rot = rotations[layer]
                    q_eval = q @ rot
                    k_eval = k @ rot
                else:
                    q_eval = q
                    k_eval = k

                k_hat = oscar2_k_dequant(k_eval)
                k_affine, coeff = affine4_reconstruct(k_eval, k_hat)
                k_sidecar_bytes = coeff.shape[0] * 8 * 2
                k_bulk_bytes = coeff.shape[0] * (2 * 2 + 128 // 4 + 128 // 8)
                kq_ref = kq_scores(q_eval, k_eval)
                kq_base = kq_scores(q_eval, k_hat)
                kq_affine = kq_scores(q_eval, k_affine)
                base_nmse = nmse(kq_ref, kq_base)
                aff_nmse = nmse(kq_ref, kq_affine)
                improvement = (base_nmse - aff_nmse) / max(base_nmse, 1e-30)
                rows.append({
                    "prompt": prompt,
                    "layer": layer,
                    "mode": mode,
                    "tokens": q_eval.shape[0],
                    "k_nmse_base": nmse(k_eval, k_hat),
                    "k_nmse_affine4": nmse(k_eval, k_affine),
                    "kq_nmse_base": base_nmse,
                    "kq_nmse_affine4": aff_nmse,
                    "kq_improvement": improvement,
                    "kq_cos_base": cosine(kq_ref, kq_base),
                    "kq_cos_affine4": cosine(kq_ref, kq_affine),
                    "coeff_abs_max": float(np.max(np.abs(coeff))),
                    "k_sidecar_bytes": k_sidecar_bytes,
                    "k_bulk_bytes": k_bulk_bytes,
                    "k_sidecar_over_k_bulk": k_sidecar_bytes / max(k_bulk_bytes, 1),
                })

    if not rows:
        raise RuntimeError(f"no Q/K rows found in {args.dump_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "affine4_kq_error.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        base = np.array([float(r["kq_nmse_base"]) for r in sub], dtype=np.float64)
        aff = np.array([float(r["kq_nmse_affine4"]) for r in sub], dtype=np.float64)
        imp = (base - aff) / np.maximum(base, 1e-30)
        summary_rows.append({
            "mode": mode,
            "rows": len(sub),
            "base_mean": float(base.mean()),
            "affine4_mean": float(aff.mean()),
            "mean_improvement": float(imp.mean()),
            "median_improvement": float(np.median(imp)),
            "worst_improvement": float(imp.min()),
            "gate": "pass" if float(imp.mean()) >= args.min_improvement else "fail",
        })

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "| mode | rows | base KQ NMSE mean | affine4 KQ NMSE mean | mean improvement | median improvement | worst improvement | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['mode']} | {row['rows']} | {row['base_mean']:.8g} | {row['affine4_mean']:.8g} | "
            f"{row['mean_improvement']:.3f} | {row['median_improvement']:.3f} | {row['worst_improvement']:.3f} | {row['gate']} |"
        )
    md = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(md)
    print(md, end="")

    if not any(row["gate"] == "pass" for row in summary_rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
