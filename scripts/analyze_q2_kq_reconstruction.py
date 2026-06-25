#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from analyze_q2_kq_softmax_dump import parse_meta
from analyze_q2_runtime_cache_dump import dequant_q2_cache, load_indices


META_RE = re.compile(r"^(?P<tensor>.+)-(?P<layer>\d+)\.(?P<idx>\d+)\.meta\.txt$")


def load(meta_path: Path) -> tuple[np.ndarray, list[int]]:
    meta = parse_meta(meta_path)
    if meta["type"] != "f32":
        raise ValueError(f"{meta_path}: expected f32, got {meta['type']}")
    ne = [int(x) for x in meta["ne"].split(",")]
    arr = np.fromfile(meta_path.with_suffix("").with_suffix(".bin"), dtype=np.float32)
    expected = int(np.prod(ne))
    if arr.size != expected:
        raise ValueError(f"{meta_path}: expected {expected} floats, got {arr.size}")
    return arr.reshape(tuple(reversed(ne))).copy(), ne


def paths_for(tensor_dir: Path, tensor: str, layer: int) -> list[Path]:
    out = []
    for meta_path in tensor_dir.glob(f"{tensor}-{layer}.*.meta.txt"):
        match = META_RE.match(meta_path.name)
        if match:
            out.append(meta_path)
    return sorted(out)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    return float(np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-30))


def scaled_nmse(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xx = x.astype(np.float64).ravel()
    yy = y.astype(np.float64).ravel()
    scale = float(np.dot(xx, yy) / (np.dot(xx, xx) + 1e-30))
    return scale, float(np.mean((scale * xx - yy) ** 2) / (np.mean(yy ** 2) + 1e-30))


def candidate_q(meta_path: Path) -> np.ndarray | None:
    arr, ne = load(meta_path)
    if ne[0] == 128 and ne[1] == 16 and ne[2] == 15:
        return arr.reshape(15, 16, 128)
    return None


def candidate_k(meta_path: Path) -> np.ndarray | None:
    arr, ne = load(meta_path)
    if ne[0] == 128 and ne[1] == 4 and ne[2] == 15:
        return arr.reshape(15, 4, 128)
    return None


def candidate_cache_k(tensor_dir: Path, layer: int, dims: int = 128) -> list[tuple[str, np.ndarray]]:
    meta_path = tensor_dir / f"cache_k_set_rows-{layer}.0.meta.txt"
    idx_path = tensor_dir / f"cache_k_set_rows-{layer}_src1.0.meta.txt"
    if not meta_path.is_file() or not idx_path.is_file():
        return []
    meta = parse_meta(meta_path)
    if meta.get("type") != "q2_0":
        return []
    ne = [int(x) for x in meta["ne"].split(",")]
    idx_meta = parse_meta(idx_path)
    indices = load_indices(idx_path.with_suffix("").with_suffix(".bin"), idx_meta)
    rows = dequant_q2_cache(meta_path.with_suffix("").with_suffix(".bin"), ne[1], dims * 4, owht=True, no_hadamard=True)
    used = rows[indices[:15]].reshape(15, 4, dims)
    return [(meta_path.name, used)]


def candidate_kq(meta_path: Path) -> np.ndarray | None:
    arr, ne = load(meta_path)
    if ne[0] == 256 and ne[1] == 15 and ne[2] == 16:
        return arr.reshape(16, 15, 256)[:, :, :15]
    return None


def reconstruct(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.empty((16, 15, 15), dtype=np.float32)
    for h in range(16):
        out[h] = q[:, h, :] @ k[:, h // 4, :].T
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether dumped kq is reconstructable from dumped Qcur/Kcur.")
    parser.add_argument("--root", type=Path, default=Path("runs/q2_kq_softmax_dump_current"))
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for variant_dir in sorted(p for p in args.root.iterdir() if (p / "tensors").is_dir()):
        tensor_dir = variant_dir / "tensors"
        for layer in range(40):
            kq_candidates = [(p, candidate_kq(p)) for p in paths_for(tensor_dir, "kq", layer)]
            kq_candidates = [(p, x) for p, x in kq_candidates if x is not None]
            q_candidates = [(p, candidate_q(p)) for p in paths_for(tensor_dir, "Qcur", layer)]
            q_candidates = [(p, x) for p, x in q_candidates if x is not None]
            k_candidates = [(p, candidate_k(p)) for p in paths_for(tensor_dir, "Kcur", layer)]
            k_candidates = [(p, x) for p, x in k_candidates if x is not None]
            cache_candidates = [(Path(name), x) for name, x in candidate_cache_k(tensor_dir, layer)]
            k_candidates.extend(cache_candidates)
            if not kq_candidates or not q_candidates or not k_candidates:
                continue

            best = None
            for kq_path, kq in kq_candidates:
                for q_path, q in q_candidates:
                    for k_path, k in k_candidates:
                        pred = reconstruct(q, k)
                        cos = cosine(pred, kq)
                        scale, err = scaled_nmse(pred, kq)
                        item = (cos, err, scale, kq_path.name, q_path.name, k_path.name)
                        if best is None or item[0] > best[0]:
                            best = item
            assert best is not None
            rows.append({
                "variant": variant_dir.name,
                "layer": str(layer),
                "best_cos": f"{best[0]:.8g}",
                "best_scaled_nmse": f"{best[1]:.8g}",
                "best_scale": f"{best[2]:.8g}",
                "kq_file": best[3],
                "q_file": best[4],
                "k_file": best[5],
            })

    if not rows:
        raise SystemExit(f"no reconstruction rows found under {args.root}")

    out_csv = args.out_csv or args.root / "kq_reconstruction.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_csv}")
    for variant in sorted({row["variant"] for row in rows}):
        vals = [float(row["best_cos"]) for row in rows if row["variant"] == variant]
        print(f"{variant}: min_best_cos={min(vals):.8g}")


if __name__ == "__main__":
    main()
