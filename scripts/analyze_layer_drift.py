#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


NAME_RE = re.compile(r"^(?P<name>.+)-(?P<layer>\d+)\.0\.meta\.txt$")


def parse_meta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def load_tensor(meta_path: Path) -> np.ndarray:
    meta = parse_meta(meta_path)
    if meta["type"] != "f32":
        raise ValueError(f"{meta_path}: expected f32, got {meta['type']}")
    bin_path = meta_path.with_name(meta_path.name.removesuffix(".meta.txt") + ".bin")
    ne = [int(x) for x in meta["ne"].split(",")]
    arr = np.fromfile(bin_path, dtype=np.float32)
    expected = int(np.prod(ne))
    if arr.size != expected:
        raise ValueError(f"{bin_path}: expected {expected} floats, got {arr.size}")
    return arr


def cosine(x: np.ndarray, ref: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    rr = ref.astype(np.float64).ravel()
    denom = np.linalg.norm(xx) * np.linalg.norm(rr) + 1e-30
    return float(np.dot(xx, rr) / denom)


def nmse(x: np.ndarray, ref: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    rr = ref.astype(np.float64).ravel()
    return float(np.mean((xx - rr) ** 2) / (np.mean(rr ** 2) + 1e-30))


def collect(tensor_dir: Path, names: set[str]) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for meta_path in tensor_dir.glob("*.meta.txt"):
        m = NAME_RE.match(meta_path.name)
        if not m:
            continue
        name = m.group("name")
        if name not in names:
            continue
        out[(name, int(m.group("layer")))] = meta_path
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/q2_logits_path_dump_current"))
    parser.add_argument("--reference", default="oscar_bf16")
    parser.add_argument("--tensors", default="__fattn__,kqv_out,kqv_out__reshaped_")
    args = parser.parse_args()

    names = {x.strip() for x in args.tensors.split(",") if x.strip()}
    variants = sorted(p.name for p in args.root.iterdir() if (p / "tensors").is_dir())
    ref_paths = collect(args.root / args.reference / "tensors", names)

    rows: list[dict[str, str]] = []
    for variant in variants:
        paths = collect(args.root / variant / "tensors", names)
        for key, ref_path in sorted(ref_paths.items()):
            if key not in paths:
                continue
            name, layer = key
            ref = load_tensor(ref_path)
            cur = load_tensor(paths[key])
            rows.append({
                "variant": variant,
                "tensor": name,
                "layer": str(layer),
                "finite": str(bool(np.isfinite(cur).all())).lower(),
                "cos_vs_ref": f"{cosine(cur, ref):.8g}",
                "nmse_vs_ref": f"{nmse(cur, ref):.8g}",
                "std": f"{float(np.std(cur)):.8g}",
                "max_abs": f"{float(np.max(np.abs(cur))):.8g}",
            })

    csv_path = args.root / "layer_drift.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.root / "layer_drift_summary.md"
    with md_path.open("w") as f:
        f.write("# Layer Drift Summary\n\n")
        f.write("| variant | tensor | worst layer | min cosine | max NMSE |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for variant in variants:
            for name in sorted(names):
                subset = [r for r in rows if r["variant"] == variant and r["tensor"] == name]
                if not subset:
                    continue
                worst = min(subset, key=lambda r: float(r["cos_vs_ref"]))
                max_nmse = max(float(r["nmse_vs_ref"]) for r in subset)
                f.write(
                    f"| {variant} | {name} | {worst['layer']} | "
                    f"{worst['cos_vs_ref']} | {max_nmse:.8g} |\n"
                )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
