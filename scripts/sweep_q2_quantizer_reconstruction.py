#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import analyze_q2_runtime_cache_dump as q2dump


MODES = {
    "plain_q2": {"owht": False, "no_hadamard": True, "clip_k": 0.0, "clip_v": 0.0},
    "owht_no_clip": {"owht": True, "no_hadamard": True, "clip_k": 0.0, "clip_v": 0.0},
    "owht_split_clip": {"owht": True, "no_hadamard": True, "clip_k": 0.96, "clip_v": 0.92},
}


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    return float(np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-30))


def load_src(dump_dir: Path, kind: str, layer: int) -> np.ndarray:
    path = dump_dir / f"cache_{kind}_set_rows-{layer}_src0.0.bin"
    meta = q2dump.parse_meta(path.with_suffix(".meta.txt"))
    return q2dump.load_f32(path, q2dump.meta_ne(meta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=Path("runs/q2_runtime_cache_dump_current/on/dump"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/q2_quantizer_reconstruction_sweep_current"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for kind, layer in q2dump.discover_layers(args.dump_dir):
        src = load_src(args.dump_dir, kind, layer)
        for mode, cfg in MODES.items():
            clip = cfg["clip_k"] if kind == "k" else cfg["clip_v"]
            recon = q2dump.q2_quantize_dequant(
                src,
                owht=bool(cfg["owht"]),
                no_hadamard=bool(cfg["no_hadamard"]),
                clip_ratio=float(clip),
            )
            rows.append({
                "kind": kind.upper(),
                "layer": str(layer),
                "mode": mode,
                "clip_ratio": f"{clip:.8g}",
                "nmse": f"{q2dump.nmse_np(src, recon):.8g}",
                "cosine": f"{cosine(src, recon):.8g}",
                "max_abs": f"{float(np.max(np.abs(recon - src))):.8g}",
                "src_std": f"{float(np.std(src)):.8g}",
                "recon_std": f"{float(np.std(recon)):.8g}",
            })

    csv_path = args.out_dir / "q2_quantizer_reconstruction_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out_dir / "q2_quantizer_reconstruction_sweep.md"
    with md_path.open("w") as f:
        f.write("# Q2 Quantizer Reconstruction Sweep\n\n")
        f.write("| kind | mode | mean NMSE | max NMSE | mean cosine | min cosine |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for kind in ("K", "V"):
            for mode in MODES:
                sub = [r for r in rows if r["kind"] == kind and r["mode"] == mode]
                nmse = np.array([float(r["nmse"]) for r in sub])
                cos = np.array([float(r["cosine"]) for r in sub])
                f.write(
                    f"| {kind} | {mode} | {float(nmse.mean()):.8g} | {float(nmse.max()):.8g} | "
                    f"{float(cos.mean()):.8g} | {float(cos.min()):.8g} |\n"
                )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
