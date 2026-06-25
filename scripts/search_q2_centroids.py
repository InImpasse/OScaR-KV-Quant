#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from compare_int2_attention_error import causal_attention, load_tensor, nmse
from compare_int2_quantizers import DEFAULT_GGUF, load_gguf_rotations


ROOT = Path(__file__).resolve().parents[1]


def q2_symmetric_dequant(
    x: np.ndarray,
    *,
    c0: float,
    c1: float,
    threshold: float,
    group_size: int,
    rotate_back: np.ndarray | None = None,
) -> np.ndarray:
    original_shape = x.shape
    grouped = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
    mean = grouped.mean(axis=-1, keepdims=True)
    centered = grouped - mean
    blocks = centered.reshape(*original_shape[:-1], original_shape[-1] // 32, 32)
    sigma = np.sqrt(np.mean(blocks * blocks, axis=-1, keepdims=True)).clip(min=1e-8)
    scaled = blocks / sigma
    abs_scaled = np.abs(scaled)
    sign = np.where(scaled < 0, -1.0, 1.0).astype(np.float32)
    deq_blocks = sigma * sign * np.where(abs_scaled < threshold, c1, c0).astype(np.float32)
    deq = (deq_blocks.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size) + mean).reshape(original_shape)
    if rotate_back is not None:
        deq = deq @ rotate_back.T
    return deq


def load_items(dump_dir: Path) -> dict[tuple[str, int, str], dict]:
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    by_key = {}
    for item in manifest["items"]:
        if item["kind"] in {"Qcur", "Kcur", "Vcur"}:
            layer = int(item["name"].split("-", 1)[1])
            by_key[(item["prompt"], layer, item["kind"])] = item
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Search q2_0 symmetric centroids/threshold for Granite OSCAR KV attention error.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/granite_fp16_qkv_calib_20260604T172700Z")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/q2_centroid_search_current")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    by_key = load_items(args.dump_dir)
    k_rotations = load_gguf_rotations(args.gguf, "k", args.max_layers)
    v_rotations = load_gguf_rotations(args.gguf, "v", args.max_layers)

    cases = []
    for prompt_idx in range(args.max_prompts):
        prompt = f"prompt_{prompt_idx}"
        for layer in range(args.max_layers):
            try:
                q_item = by_key[(prompt, layer, "Qcur")]
                k_item = by_key[(prompt, layer, "Kcur")]
                v_item = by_key[(prompt, layer, "Vcur")]
            except KeyError:
                continue
            q = load_tensor(args.dump_dir, q_item)
            k = load_tensor(args.dump_dir, k_item)
            v = load_tensor(args.dump_dir, v_item)
            cases.append({
                "prompt": prompt,
                "layer": layer,
                "q": q,
                "k_rot": k @ k_rotations[layer],
                "v_rot": v @ v_rotations[layer],
                "k_rot_back": k_rotations[layer],
                "v_rot_back": v_rotations[layer],
                "ref": causal_attention(q, k, v),
            })

    if not cases:
        raise SystemExit("no QKV dump cases found")

    # Include current Lloyd-Max parameters, then sweep nearby and lower-magnitude
    # levels. K-side attention error is the gate, so the grid is deliberately
    # biased toward flatter centroids that reduce KQ overconfidence.
    c0_values = np.unique(np.concatenate([
        np.array([0.9816], dtype=np.float32),
        np.linspace(0.62, 1.12, 11, dtype=np.float32),
    ]))
    c1_values = np.unique(np.concatenate([
        np.array([0.4528], dtype=np.float32),
        np.linspace(0.20, 0.58, 10, dtype=np.float32),
    ]))
    t_values = np.unique(np.concatenate([
        np.array([0.6745], dtype=np.float32),
        np.linspace(0.38, 0.88, 11, dtype=np.float32),
    ]))

    rows = []
    total = 0
    for c0 in c0_values:
        for c1 in c1_values:
            if c1 >= c0:
                continue
            for threshold in t_values:
                total += 1
                vals = []
                k_only_vals = []
                v_only_vals = []
                for case in cases:
                    kq = q2_symmetric_dequant(
                        case["k_rot"],
                        c0=float(c0),
                        c1=float(c1),
                        threshold=float(threshold),
                        group_size=128,
                        rotate_back=case["k_rot_back"],
                    )
                    vq = q2_symmetric_dequant(
                        case["v_rot"],
                        c0=float(c0),
                        c1=float(c1),
                        threshold=float(threshold),
                        group_size=128,
                        rotate_back=case["v_rot_back"],
                    )
                    both = causal_attention(case["q"], kq, vq)
                    k_only = causal_attention(case["q"], kq, case["v_rot"] @ case["v_rot_back"].T)
                    v_only = causal_attention(case["q"], case["k_rot"] @ case["k_rot_back"].T, vq)
                    vals.append(nmse(case["ref"], both))
                    k_only_vals.append(nmse(case["ref"], k_only))
                    v_only_vals.append(nmse(case["ref"], v_only))
                rows.append({
                    "c0": float(c0),
                    "c1": float(c1),
                    "threshold": float(threshold),
                    "attn_nmse_mean": float(np.mean(vals)),
                    "attn_nmse_median": float(np.median(vals)),
                    "attn_nmse_max": float(np.max(vals)),
                    "k_only_nmse_mean": float(np.mean(k_only_vals)),
                    "v_only_nmse_mean": float(np.mean(v_only_vals)),
                })

    rows.sort(key=lambda r: (r["attn_nmse_mean"], r["attn_nmse_max"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "centroid_search.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top = rows[: args.top_k]
    md = [
        f"# Q2 Centroid Search",
        "",
        f"cases={len(cases)}, candidates={total}",
        "",
        "| rank | c0 | c1 | threshold | mean attn NMSE | max attn NMSE | K-only mean | V-only mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(top, start=1):
        md.append(
            f"| {i} | {row['c0']:.6f} | {row['c1']:.6f} | {row['threshold']:.6f} | "
            f"{row['attn_nmse_mean']:.6f} | {row['attn_nmse_max']:.6f} | "
            f"{row['k_only_nmse_mean']:.6f} | {row['v_only_nmse_mean']:.6f} |"
        )
    (args.out_dir / "centroid_search.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
