#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from compare_int2_quantizers import DEFAULT_GGUF, int2_asym_dequant, load_gguf_rotations, q2_0_dequant, turbo2_dequant, turbo3_dequant


ROOT = Path(__file__).resolve().parents[1]


def load_tensor(run_dir: Path, item: dict) -> np.ndarray:
    raw = np.fromfile(run_dir / item["bin"], dtype=np.float32)
    ne = item["ne"]
    expected = int(np.prod(ne))
    if raw.size != expected:
        raise ValueError(f"{item['bin']} has {raw.size} floats, expected {expected}")
    arr = raw.reshape((ne[3], ne[2], ne[1], ne[0]))
    return arr.reshape(ne[2], ne[1], ne[0]).astype(np.float32, copy=True)


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    err = (y - x).astype(np.float32)
    denom = max(float((x.astype(np.float32) * x.astype(np.float32)).mean()), 1e-12)
    return float((err * err).mean() / denom)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q: T,H,D, k/v: T,HKV,D. Granite 1B uses GQA, H is a multiple of HKV.
    tokens, n_heads, dim = q.shape
    kv_heads = k.shape[1]
    group = n_heads // kv_heads
    out = np.empty_like(q)
    scale = 1.0 / math.sqrt(dim)
    causal = np.triu(np.ones((tokens, tokens), dtype=bool), k=1)
    for h in range(n_heads):
        hkv = h // group
        scores = (q[:, h, :] @ k[:, hkv, :].T) * scale
        scores = np.where(causal, -np.inf, scores)
        probs = softmax(scores)
        out[:, h, :] = probs @ v[:, hkv, :]
    return out


def quantize_pair(k: np.ndarray, v: np.ndarray, mode: str, k_rot: np.ndarray, v_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == "q2":
        return q2_0_dequant(k), q2_0_dequant(v)
    if mode == "q2_rot":
        return q2_0_dequant(k @ k_rot) @ k_rot.T, q2_0_dequant(v @ v_rot) @ v_rot.T
    if mode == "q2_oscar_rot":
        return q2_0_dequant(k @ k_rot, group_size=128, clip_ratio=0.96) @ k_rot.T, q2_0_dequant(v @ v_rot, group_size=128, clip_ratio=0.92) @ v_rot.T
    if mode == "q2_oscar_konly":
        return q2_0_dequant(k @ k_rot, group_size=128, clip_ratio=0.96) @ k_rot.T, v
    if mode == "q2_oscar_vonly":
        return k, q2_0_dequant(v @ v_rot, group_size=128, clip_ratio=0.92) @ v_rot.T
    if mode == "asym":
        return int2_asym_dequant(k), int2_asym_dequant(v)
    if mode == "asym_rot":
        return int2_asym_dequant(k @ k_rot) @ k_rot.T, int2_asym_dequant(v @ v_rot) @ v_rot.T
    if mode == "turbo2":
        return turbo2_dequant(k), turbo2_dequant(v)
    if mode == "turbo2_rot":
        return turbo2_dequant(k @ k_rot) @ k_rot.T, turbo2_dequant(v @ v_rot) @ v_rot.T
    if mode == "turbo3":
        return turbo3_dequant(k), turbo3_dequant(v)
    if mode == "turbo3_rot":
        return turbo3_dequant(k @ k_rot) @ k_rot.T, turbo3_dequant(v @ v_rot) @ v_rot.T
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attention-output error for q2_0 and asymmetric int2 on dumped Granite QKV tensors.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/granite_fp16_qkv_calib_20260604T172700Z")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/int2_attention_error_current")
    args = parser.parse_args()

    manifest = json.loads((args.dump_dir / "manifest.json").read_text())
    by_key = {}
    for item in manifest["items"]:
        if item["kind"] in {"Qcur", "Kcur", "Vcur"}:
            layer = int(item["name"].split("-", 1)[1])
            by_key[(item["prompt"], layer, item["kind"])] = item
    k_rotations = load_gguf_rotations(args.gguf, "k", args.max_layers)
    v_rotations = load_gguf_rotations(args.gguf, "v", args.max_layers)

    rows = []
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
            ref = causal_attention(q, k, v)
            k_rot = k_rotations[layer]
            v_rot = v_rotations[layer]
            row = {"prompt": prompt, "layer": layer, "tokens": q.shape[0]}
            for mode in (
                "q2",
                "q2_rot",
                "q2_oscar_rot",
                "q2_oscar_konly",
                "q2_oscar_vonly",
                "asym",
                "asym_rot",
                "turbo2",
                "turbo2_rot",
                "turbo3",
                "turbo3_rot",
            ):
                kq, vq = quantize_pair(k, v, mode, k_rot, v_rot)
                out = causal_attention(q, kq, vq)
                row[f"{mode}_attn_nmse"] = nmse(ref, out)
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "attention_error.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"rows": len(rows)}
    for key in (
        "q2_attn_nmse",
        "q2_rot_attn_nmse",
        "q2_oscar_rot_attn_nmse",
        "q2_oscar_konly_attn_nmse",
        "q2_oscar_vonly_attn_nmse",
        "asym_attn_nmse",
        "asym_rot_attn_nmse",
        "turbo2_attn_nmse",
        "turbo2_rot_attn_nmse",
        "turbo3_attn_nmse",
        "turbo3_rot_attn_nmse",
    ):
        vals = np.array([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_median"] = float(np.median(vals))
        summary[f"{key}_max"] = float(vals.max())
    summary["q2_rot_vs_plain_mean_ratio"] = summary["q2_rot_attn_nmse_mean"] / max(summary["q2_attn_nmse_mean"], 1e-12)
    summary["q2_oscar_rot_vs_plain_mean_ratio"] = summary["q2_oscar_rot_attn_nmse_mean"] / max(summary["q2_attn_nmse_mean"], 1e-12)
    summary["q2_oscar_konly_vs_vonly_mean_ratio"] = summary["q2_oscar_konly_attn_nmse_mean"] / max(summary["q2_oscar_vonly_attn_nmse_mean"], 1e-12)
    summary["asym_rot_vs_plain_mean_ratio"] = summary["asym_rot_attn_nmse_mean"] / max(summary["asym_attn_nmse_mean"], 1e-12)
    summary["q2rot_vs_asymrot_mean_ratio"] = summary["q2_rot_attn_nmse_mean"] / max(summary["asym_rot_attn_nmse_mean"], 1e-12)
    summary["turbo2_rot_vs_plain_mean_ratio"] = summary["turbo2_rot_attn_nmse_mean"] / max(summary["turbo2_attn_nmse_mean"], 1e-12)
    summary["turbo3_rot_vs_plain_mean_ratio"] = summary["turbo3_rot_attn_nmse_mean"] / max(summary["turbo3_attn_nmse_mean"], 1e-12)
    summary["turbo3rot_vs_q2rot_mean_ratio"] = summary["turbo3_rot_attn_nmse_mean"] / max(summary["q2_rot_attn_nmse_mean"], 1e-12)

    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    md = [
        "| rows | q2+rot mean | q2 OSCAR rot mean | q2 OSCAR K-only | q2 OSCAR V-only | asym+rot mean | turbo2+rot mean | turbo3+rot mean | turbo3rot/q2rot |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['rows']} | "
        f"{summary['q2_rot_attn_nmse_mean']:.6f} | {summary['q2_oscar_rot_attn_nmse_mean']:.6f} | "
        f"{summary['q2_oscar_konly_attn_nmse_mean']:.6f} | {summary['q2_oscar_vonly_attn_nmse_mean']:.6f} | "
        f"{summary['asym_rot_attn_nmse_mean']:.6f} | {summary['turbo2_rot_attn_nmse_mean']:.6f} | "
        f"{summary['turbo3_rot_attn_nmse_mean']:.6f} | {summary['turbo3rot_vs_q2rot_mean_ratio']:.3f} |",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")
    (args.out_dir / "README.md").write_text(
        "Offline causal attention-output NMSE on dumped Granite Q/K/V tensors. "
        "Uses exact Q, quantized/dequantized K/V, and GQA head mapping.\n"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
