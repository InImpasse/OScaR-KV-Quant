#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import analyze_q2_runtime_cache_dump as cache_dump


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    return float(np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-30))


def compare_one(dump_dir: Path, cache_kind: str, layer: int, *, owht: bool, no_hadamard: bool) -> dict[str, str]:
    src_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}_src0.0.bin"
    src_meta = cache_dump.parse_meta(src_path.with_suffix(".meta.txt"))
    src = cache_dump.load_f32(src_path, cache_dump.meta_ne(src_meta))

    idx_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}_src1.0.bin"
    idx_meta = cache_dump.parse_meta(idx_path.with_suffix(".meta.txt"))
    indices = cache_dump.load_indices(idx_path, idx_meta)

    cache_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}.0.bin"
    cache_meta = cache_dump.parse_meta(cache_path.with_suffix(".meta.txt"))
    cache_ne = cache_dump.meta_ne(cache_meta)
    tokens = src.shape[0]
    dims = src.shape[1]
    cache_rows = cache_dump.dequant_q2_cache(cache_path, cache_ne[1], dims, owht=owht, no_hadamard=no_hadamard)
    actual = cache_rows[indices[:tokens]]

    diff = actual - src
    return {
        "kind": cache_kind.upper(),
        "layer": str(layer),
        "tokens": str(tokens),
        "dims": str(dims),
        "nmse_recon_vs_src": f"{cache_dump.nmse_np(src, actual):.8g}",
        "cos_recon_vs_src": f"{cosine(src, actual):.8g}",
        "max_abs_recon_vs_src": f"{float(np.max(np.abs(diff))):.8g}",
        "src_std": f"{float(np.std(src)):.8g}",
        "recon_std": f"{float(np.std(actual)):.8g}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=Path("runs/q2_runtime_cache_dump_current/on/dump"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/q2_runtime_cache_dump_current/on"))
    parser.add_argument("--owht", action="store_true")
    parser.add_argument("--no-hadamard", action="store_true")
    args = parser.parse_args()

    rows = [
        compare_one(args.dump_dir, kind, layer, owht=args.owht, no_hadamard=args.no_hadamard)
        for kind, layer in cache_dump.discover_layers(args.dump_dir)
    ]
    if not rows:
        raise SystemExit(f"no q2 cache dump rows found in {args.dump_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "cache_reconstruction_error.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out_dir / "cache_reconstruction_error.md"
    with md_path.open("w") as f:
        f.write("# Q2 Cache Reconstruction Error\n\n")
        f.write("| kind | worst layer | max NMSE | min cosine | max abs |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for kind in ("K", "V"):
            sub = [r for r in rows if r["kind"] == kind]
            worst_nmse = max(sub, key=lambda r: float(r["nmse_recon_vs_src"]))
            worst_cos = min(sub, key=lambda r: float(r["cos_recon_vs_src"]))
            max_abs = max(float(r["max_abs_recon_vs_src"]) for r in sub)
            f.write(
                f"| {kind} | {worst_nmse['layer']} | {worst_nmse['nmse_recon_vs_src']} | "
                f"{worst_cos['cos_recon_vs_src']} | {max_abs:.8g} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
