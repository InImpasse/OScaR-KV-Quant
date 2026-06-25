#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_q2_kq_softmax_dump import parse_meta
from analyze_q2_runtime_cache_dump import dequant_q2_cache, load_indices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TENSOR_DIR = ROOT / "runs/q2_kq_softmax_dump_current/oscar_kq2_vbf16/tensors"


def load_f32(meta_path: Path) -> tuple[np.ndarray, list[int]]:
    meta = parse_meta(meta_path)
    if meta["type"] != "f32":
        raise ValueError(f"{meta_path}: expected f32, got {meta['type']}")
    ne = [int(x) for x in meta["ne"].split(",")]
    raw = np.fromfile(meta_path.with_suffix("").with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(ne))
    if raw.size != expected:
        raise ValueError(f"{meta_path}: expected {expected} floats, got {raw.size}")
    return raw.reshape(tuple(reversed(ne))).copy(), ne


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    return float(np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-30))


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    err = y.astype(np.float64) - x.astype(np.float64)
    denom = max(float(np.mean(x.astype(np.float64) ** 2)), 1e-30)
    return float(np.mean(err * err) / denom)


def load_q(meta_path: Path) -> np.ndarray:
    arr, ne = load_f32(meta_path)
    if ne[:3] != [128, 16, 15]:
        raise ValueError(f"{meta_path}: unexpected Q shape ne={ne}")
    return arr.reshape(15, 16, 128)


def load_k(meta_path: Path) -> np.ndarray:
    arr, ne = load_f32(meta_path)
    if ne[:3] != [128, 4, 15]:
        raise ValueError(f"{meta_path}: unexpected K shape ne={ne}")
    return arr.reshape(15, 4, 128)


def load_cache_k(tensor_dir: Path, layer: int, *, owht: bool, no_hadamard: bool) -> np.ndarray:
    meta_path = tensor_dir / f"cache_k_set_rows-{layer}.0.meta.txt"
    idx_path = tensor_dir / f"cache_k_set_rows-{layer}_src1.0.meta.txt"
    if not meta_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError(f"missing q2 cache dump for layer {layer}")
    meta = parse_meta(meta_path)
    if meta["type"] != "q2_0":
        raise ValueError(f"{meta_path}: expected q2_0, got {meta['type']}")
    ne = [int(x) for x in meta["ne"].split(",")]
    if ne[0] != 512:
        raise ValueError(f"{meta_path}: expected packed K width 512, got ne={ne}")

    idx_meta = parse_meta(idx_path)
    indices = load_indices(idx_path.with_suffix("").with_suffix(".bin"), idx_meta)
    rows = dequant_q2_cache(meta_path.with_suffix("").with_suffix(".bin"), ne[1], 512, owht=owht, no_hadamard=no_hadamard)
    return rows[indices[:15]].reshape(15, 4, 128)


def kq(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.empty((16, 15, 15), dtype=np.float32)
    for h in range(16):
        out[h] = q[:, h, :] @ k[:, h // 4, :].T
    return out


def analyze_layer(tensor_dir: Path, layer: int, *, owht: bool, no_hadamard: bool) -> dict[str, str]:
    q = load_q(tensor_dir / f"Qcur_rot-{layer}.0.meta.txt")
    k_f32 = load_k(tensor_dir / f"Kcur_rot-{layer}.0.meta.txt")
    k_q2 = load_cache_k(tensor_dir, layer, owht=owht, no_hadamard=no_hadamard)

    kq_f32 = kq(q, k_f32)
    kq_q2 = kq(q, k_q2)

    return {
        "layer": str(layer),
        "owht": str(int(owht)),
        "no_hadamard": str(int(no_hadamard)),
        "k_cos": f"{cosine(k_f32, k_q2):.8g}",
        "k_nmse": f"{nmse(k_f32, k_q2):.8g}",
        "kq_cos": f"{cosine(kq_f32, kq_q2):.8g}",
        "kq_nmse": f"{nmse(kq_f32, kq_q2):.8g}",
        "kq_abs_max": f"{float(np.max(np.abs(kq_q2 - kq_f32))):.8g}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare f32 KQ against KQ reconstructed from q2 cache rows.")
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/q2_cache_kq_error_current")
    parser.add_argument("--owht", action="store_true", default=True)
    parser.add_argument("--plain", dest="owht", action="store_false")
    parser.add_argument("--no-hadamard", action="store_true", default=True)
    parser.add_argument("--with-hadamard", dest="no_hadamard", action="store_false")
    parser.add_argument("--max-layers", type=int, default=40)
    args = parser.parse_args()

    rows = [analyze_layer(args.tensor_dir, layer, owht=args.owht, no_hadamard=args.no_hadamard) for layer in range(args.max_layers)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "cache_kq_error.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    min_k_cos = min(float(row["k_cos"]) for row in rows)
    max_k_nmse = max(float(row["k_nmse"]) for row in rows)
    min_kq_cos = min(float(row["kq_cos"]) for row in rows)
    max_kq_nmse = max(float(row["kq_nmse"]) for row in rows)
    worst = min(rows, key=lambda row: float(row["kq_cos"]))
    md = [
        "| layers | OWHT | no-Hadamard | min K cosine | max K NMSE | min KQ cosine | max KQ NMSE | worst KQ layer |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {int(args.owht)} | {int(args.no_hadamard)} | {min_k_cos:.8g} | {max_k_nmse:.8g} | {min_kq_cos:.8g} | {max_kq_nmse:.8g} | {worst['layer']} |",
    ]
    md_path = args.out_dir / "cache_kq_error.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(md[-1])


if __name__ == "__main__":
    main()
