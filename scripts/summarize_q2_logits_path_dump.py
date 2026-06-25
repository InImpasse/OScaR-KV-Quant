#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_TENSORS = ("result_norm", "result_output", "__fattn__-39", "kqv_out-39")


def parse_meta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def load_tensor(dump_dir: Path, name: str) -> np.ndarray:
    meta = parse_meta(dump_dir / f"{name}.0.meta.txt")
    dtype = meta["type"]
    if dtype != "f32":
        raise ValueError(f"{dump_dir}/{name}: expected f32, got {dtype}")
    shape = tuple(int(x) for x in meta["ne"].split(","))
    ne = [x for x in shape if x != 1]
    arr = np.fromfile(dump_dir / f"{name}.0.bin", dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{dump_dir}/{name}: expected {expected} floats, got {arr.size}")
    return arr.reshape(tuple(reversed(ne)) if len(ne) > 1 else (arr.size,))


def nmse(x: np.ndarray, ref: np.ndarray) -> float:
    diff = x.astype(np.float64).ravel() - ref.astype(np.float64).ravel()
    denom = np.mean(ref.astype(np.float64).ravel() ** 2) + 1e-30
    return float(np.mean(diff ** 2) / denom)


def cosine(x: np.ndarray, ref: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    rr = ref.astype(np.float64).ravel()
    denom = np.linalg.norm(xx) * np.linalg.norm(rr) + 1e-30
    return float(np.dot(xx, rr) / denom)


def top_overlap(x: np.ndarray, ref: np.ndarray, k: int) -> float:
    if x.ndim != 1:
        return float("nan")
    top_x = set(np.argpartition(x, -k)[-k:].tolist())
    top_ref = set(np.argpartition(ref, -k)[-k:].tolist())
    return len(top_x & top_ref) / float(k)


def top_ids(x: np.ndarray, k: int) -> str:
    if x.ndim != 1:
        return ""
    order = np.argsort(x)[-k:][::-1]
    return " ".join(str(int(i)) for i in order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/q2_logits_path_dump_current"))
    parser.add_argument("--reference", default="oscar_bf16")
    parser.add_argument("--tensors", default=",".join(DEFAULT_TENSORS))
    args = parser.parse_args()
    tensors = tuple(t.strip() for t in args.tensors.split(",") if t.strip())

    variants = sorted(p.name for p in args.root.iterdir() if (p / "tensors").is_dir())
    if args.reference not in variants:
        raise SystemExit(f"missing reference variant {args.reference}")

    ref = {name: load_tensor(args.root / args.reference / "tensors", name) for name in tensors}

    rows: list[dict[str, str]] = []
    for variant in variants:
        for name in tensors:
            arr = load_tensor(args.root / variant / "tensors", name)
            row = {
                "variant": variant,
                "tensor": name,
                "shape": "x".join(str(x) for x in arr.shape),
                "finite": str(bool(np.isfinite(arr).all())).lower(),
                "mean": f"{float(np.mean(arr)):.8g}",
                "std": f"{float(np.std(arr)):.8g}",
                "max_abs": f"{float(np.max(np.abs(arr))):.8g}",
                "nmse_vs_oscar_bf16": "0" if variant == args.reference else f"{nmse(arr, ref[name]):.8g}",
                "cos_vs_oscar_bf16": "1" if variant == args.reference else f"{cosine(arr, ref[name]):.8g}",
                "top10_overlap_vs_oscar_bf16": f"{top_overlap(arr, ref[name], 10):.8g}",
                "top5_ids": top_ids(arr, 5),
            }
            rows.append(row)

    csv_path = args.root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.root / "summary.md"
    interesting = [r for r in rows if r["tensor"] in ("result_output", "result_norm")]
    with md_path.open("w") as f:
        f.write("# Q2 Logits Path Dump Summary\n\n")
        f.write("| variant | tensor | finite | NMSE vs oscar BF16 | cosine | top10 overlap | top5 ids |\n")
        f.write("|---|---|---:|---:|---:|---:|---|\n")
        for r in interesting:
            f.write(
                f"| {r['variant']} | {r['tensor']} | {r['finite']} | "
                f"{r['nmse_vs_oscar_bf16']} | {r['cos_vs_oscar_bf16']} | "
                f"{r['top10_overlap_vs_oscar_bf16']} | {r['top5_ids']} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
