#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_int2_attention_error import causal_attention, load_tensor, nmse
from compare_int2_quantizers import DEFAULT_GGUF, load_gguf_rotations
from search_q2_centroids import q2_symmetric_dequant


ROOT = Path(__file__).resolve().parents[1]


def load_items(dump_dir: Path) -> dict[tuple[str, int, str], dict]:
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    by_key = {}
    for item in manifest["items"]:
        if item["kind"] in {"Qcur", "Kcur", "Vcur"}:
            layer = int(item["name"].split("-", 1)[1])
            by_key[(item["prompt"], layer, item["kind"])] = item
    return by_key


def param_grid() -> list[tuple[float, float, float]]:
    values = {(0.9816, 0.4528, 0.6745)}
    for c0 in np.linspace(0.62, 1.02, 9):
        for c1 in np.linspace(0.16, 0.46, 7):
            if c1 >= c0:
                continue
            for threshold in np.linspace(0.58, 0.92, 8):
                values.add((float(c0), float(c1), float(threshold)))
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search separate q2_0 K/V symmetric centroids for OSCAR KV attention error.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/granite_fp16_qkv_calib_20260604T172700Z")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/q2_kv_split_centroid_search_current")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=24)
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
                "q": q,
                "k_rot": k @ k_rotations[layer],
                "v_rot": v @ v_rotations[layer],
                "k_rot_back": k_rotations[layer],
                "v_rot_back": v_rotations[layer],
                "ref": causal_attention(q, k, v),
            })
    if not cases:
        raise SystemExit("no QKV dump cases found")

    grid = param_grid()
    candidates = []
    # First rank K and V independently against exact counterpart to keep the
    # full K/V product search small and deterministic.
    for c0, c1, threshold in grid:
        k_vals = []
        v_vals = []
        for case in cases:
            kq = q2_symmetric_dequant(case["k_rot"], c0=c0, c1=c1, threshold=threshold, group_size=128, rotate_back=case["k_rot_back"])
            vq = q2_symmetric_dequant(case["v_rot"], c0=c0, c1=c1, threshold=threshold, group_size=128, rotate_back=case["v_rot_back"])
            k_vals.append(nmse(case["ref"], causal_attention(case["q"], kq, case["v_rot"] @ case["v_rot_back"].T)))
            v_vals.append(nmse(case["ref"], causal_attention(case["q"], case["k_rot"] @ case["k_rot_back"].T, vq)))
        candidates.append({
            "c0": c0,
            "c1": c1,
            "threshold": threshold,
            "k_only": float(np.mean(k_vals)),
            "v_only": float(np.mean(v_vals)),
        })

    top_k_params = sorted(candidates, key=lambda r: r["k_only"])[: args.top_k]
    top_v_params = sorted(candidates, key=lambda r: r["v_only"])[: args.top_k]

    rows = []
    for kp in top_k_params:
        for vp in top_v_params:
            vals = []
            for case in cases:
                kq = q2_symmetric_dequant(case["k_rot"], c0=kp["c0"], c1=kp["c1"], threshold=kp["threshold"], group_size=128, rotate_back=case["k_rot_back"])
                vq = q2_symmetric_dequant(case["v_rot"], c0=vp["c0"], c1=vp["c1"], threshold=vp["threshold"], group_size=128, rotate_back=case["v_rot_back"])
                vals.append(nmse(case["ref"], causal_attention(case["q"], kq, vq)))
            rows.append({
                "k_c0": kp["c0"],
                "k_c1": kp["c1"],
                "k_threshold": kp["threshold"],
                "v_c0": vp["c0"],
                "v_c1": vp["c1"],
                "v_threshold": vp["threshold"],
                "attn_nmse_mean": float(np.mean(vals)),
                "attn_nmse_median": float(np.median(vals)),
                "attn_nmse_max": float(np.max(vals)),
                "k_only_mean": kp["k_only"],
                "v_only_mean": vp["v_only"],
            })
    rows.sort(key=lambda r: (r["attn_nmse_mean"], r["attn_nmse_max"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "kv_split_centroid_search.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Q2 K/V Split Centroid Search",
        "",
        f"cases={len(cases)}, scalar_candidates={len(grid)}, combined_candidates={len(rows)}",
        "",
        "| rank | K c0 | K c1 | K t | V c0 | V c1 | V t | mean attn NMSE | max attn NMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[: args.top_k], start=1):
        md.append(
            f"| {i} | {row['k_c0']:.6f} | {row['k_c1']:.6f} | {row['k_threshold']:.6f} | "
            f"{row['v_c0']:.6f} | {row['v_c1']:.6f} | {row['v_threshold']:.6f} | "
            f"{row['attn_nmse_mean']:.6f} | {row['attn_nmse_max']:.6f} |"
        )
    (args.out_dir / "kv_split_centroid_search.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
