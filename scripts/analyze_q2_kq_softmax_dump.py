#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


META_RE = re.compile(r"^(?P<tensor>.+)-(?P<layer>\d+)\.(?P<pass_idx>\d+)\.meta\.txt$")


def parse_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def load_tensor(meta_path: Path) -> np.ndarray:
    meta = parse_meta(meta_path)
    if meta["type"] != "f32":
        raise ValueError(f"{meta_path}: expected f32, got {meta['type']}")
    ne = [int(x) for x in meta["ne"].split(",")]
    arr = np.fromfile(meta_path.with_suffix("").with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(ne))
    if arr.size != expected:
        raise ValueError(f"{meta_path}: expected {expected} floats, got {arr.size}")
    return arr


def classify_pass(meta_path: Path) -> str:
    meta = parse_meta(meta_path)
    ne = [int(x) for x in meta["ne"].split(",")]
    return "decode" if ne[1] <= 2 else "prefill"


def collect(tensor_dir: Path, names: set[str]) -> dict[tuple[str, int, int], Path]:
    out: dict[tuple[str, int, int], Path] = {}
    for meta_path in tensor_dir.glob("*.meta.txt"):
        match = META_RE.match(meta_path.name)
        if not match:
            continue
        tensor = match.group("tensor")
        if tensor not in names:
            continue
        out[(tensor, int(match.group("layer")), int(match.group("pass_idx")))] = meta_path
    return out


def cosine(x: np.ndarray, ref: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    rr = ref.astype(np.float64).ravel()
    return float(np.dot(xx, rr) / (np.linalg.norm(xx) * np.linalg.norm(rr) + 1e-30))


def nmse(x: np.ndarray, ref: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    rr = ref.astype(np.float64).ravel()
    return float(np.mean((xx - rr) ** 2) / (np.mean(rr ** 2) + 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze runtime non-FA KQ/softmax dumps against oscar BF16.")
    parser.add_argument("--root", type=Path, default=Path("runs/q2_kq_softmax_dump_current"))
    parser.add_argument("--reference", default="oscar_bf16")
    parser.add_argument("--tensors", default="kq,kq_soft_max,kqv_out")
    args = parser.parse_args()

    names = {name.strip() for name in args.tensors.split(",") if name.strip()}
    variants = sorted(p.name for p in args.root.iterdir() if (p / "tensors").is_dir())
    ref_paths = collect(args.root / args.reference / "tensors", names)

    rows: list[dict[str, str]] = []
    for variant in variants:
        paths = collect(args.root / variant / "tensors", names)
        for key, ref_path in sorted(ref_paths.items()):
            if key not in paths:
                continue
            tensor, layer, pass_idx = key
            cur = load_tensor(paths[key])
            ref = load_tensor(ref_path)
            rows.append({
                "variant": variant,
                "tensor": tensor,
                "layer": str(layer),
                "pass": classify_pass(paths[key]),
                "finite": str(bool(np.isfinite(cur).all())).lower(),
                "cos_vs_bf16": "1" if variant == args.reference else f"{cosine(cur, ref):.8g}",
                "nmse_vs_bf16": "0" if variant == args.reference else f"{nmse(cur, ref):.8g}",
                "std": f"{float(np.std(cur)):.8g}",
                "max_abs": f"{float(np.max(np.abs(cur))):.8g}",
            })

    if not rows:
        raise SystemExit(f"no tensors found under {args.root}")

    csv_path = args.root / "kq_softmax_drift.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.root / "kq_softmax_drift.md"
    with md_path.open("w") as f:
        f.write("# Runtime KQ / Softmax Drift\n\n")
        f.write("| variant | tensor | pass | worst layer | min cosine | max NMSE |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for variant in variants:
            if variant == args.reference:
                continue
            for tensor in sorted(names):
                for pass_kind in ("prefill", "decode"):
                    subset = [r for r in rows if r["variant"] == variant and r["tensor"] == tensor and r["pass"] == pass_kind]
                    if not subset:
                        continue
                    worst = min(subset, key=lambda r: float(r["cos_vs_bf16"]))
                    max_nmse = max(float(r["nmse_vs_bf16"]) for r in subset)
                    f.write(
                        f"| {variant} | {tensor} | {pass_kind} | {worst['layer']} | "
                        f"{worst['cos_vs_bf16']} | {max_nmse:.8g} |\n"
                    )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
