#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Q2_CENTROIDS = np.array([-0.9816, -0.4528, 0.4528, 0.9816], dtype=np.float32)
Q2_THRESHOLDS = (-0.6745, 0.0, 0.6745)


def nmse_np(x: np.ndarray, x_hat: np.ndarray) -> float:
    err = x_hat.astype(np.float32) - x.astype(np.float32)
    denom = np.maximum(np.mean(x.astype(np.float32) * x.astype(np.float32)), 1e-12)
    return float(np.mean(err * err) / denom)


def parse_meta(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        out[key] = value
    return out


def meta_ne(meta: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in meta["ne"].split(","))  # type: ignore[return-value]


def load_f32(path: Path, ne: tuple[int, int, int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(ne))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} floats, expected {expected}")
    return raw.reshape((ne[3], ne[2], ne[1], ne[0])).reshape(-1, ne[0]).copy()


def load_indices(path: Path, meta: dict[str, str]) -> np.ndarray:
    ne = meta_ne(meta)
    dtype_name = meta["type"]
    if dtype_name == "i32":
        dtype = np.int32
    elif dtype_name == "i64":
        dtype = np.int64
    else:
        raise ValueError(f"{path} has unsupported index type {dtype_name}")
    raw = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(ne))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} indices, expected {expected}")
    return raw.astype(np.int64)


def hadamard_ortho(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.reshape(-1, h * 2)
        a = y[:, :h].copy()
        b = y[:, h:].copy()
        y[:, :h] = a + b
        y[:, h:] = a - b
        y = y.reshape(*x.shape)
        h <<= 1
    return y * np.float32(1.0 / np.sqrt(n))


def clip_group(x: np.ndarray, clip_ratio: float) -> np.ndarray:
    if not (0.0 < clip_ratio < 1.0):
        return x
    y = x.copy()
    idx = int(clip_ratio * y.shape[-1])
    idx = min(idx, y.shape[-1] - 1)
    thr = np.sort(np.abs(y), axis=-1)[..., idx:idx + 1]
    return np.clip(y, -thr, thr)


def q2_quantize_dequant(src: np.ndarray, *, owht: bool, no_hadamard: bool, clip_ratio: float) -> np.ndarray:
    if src.shape[-1] % 32:
        raise ValueError(f"last dim must be divisible by 32, got {src.shape[-1]}")

    group_size = 128 if owht and src.shape[-1] >= 128 else 32
    if src.shape[-1] % group_size:
        raise ValueError(f"last dim must be divisible by group size {group_size}, got {src.shape[-1]}")

    groups = src.reshape(src.shape[0], src.shape[1] // group_size, group_size)
    mean = groups.mean(axis=-1, keepdims=True)
    tmp = groups - mean

    if owht and not no_hadamard:
        tmp = hadamard_ortho(tmp)

    if owht:
        tmp = clip_group(tmp, clip_ratio)

    blocks = tmp.reshape(src.shape[0], src.shape[1] // 32, 32)
    block_mean = np.zeros((*blocks.shape[:-1], 1), dtype=np.float32)
    if not owht:
        block_mean = blocks.mean(axis=-1, keepdims=True)
        blocks = blocks - block_mean

    sigma = np.sqrt(np.mean(blocks * blocks, axis=-1, keepdims=True)).clip(min=1e-8)
    scaled = blocks / sigma
    q = np.zeros_like(scaled, dtype=np.int64)
    q += scaled >= Q2_THRESHOLDS[0]
    q += scaled >= Q2_THRESHOLDS[1]
    q += scaled >= Q2_THRESHOLDS[2]
    deq_blocks = sigma * Q2_CENTROIDS[q]

    if not owht:
        return (deq_blocks + block_mean).reshape(src.shape)

    deq_groups = deq_blocks.reshape(src.shape[0], src.shape[1] // group_size, group_size)
    if not no_hadamard:
        deq_groups = hadamard_ortho(deq_groups)
    return (deq_groups + mean).reshape(src.shape)


def dequant_q2_cache(path: Path, rows: int, dims: int, *, owht: bool, no_hadamard: bool) -> np.ndarray:
    row_size = (dims // 32) * 12
    raw = path.read_bytes()
    if len(raw) < rows * row_size:
        raise ValueError(f"{path} too small for {rows} rows x {dims} dims")

    decoded_blocks = np.empty((rows, dims // 32, 32), dtype=np.float32)
    means = np.empty((rows, dims // 32, 1), dtype=np.float32)

    for row in range(rows):
        base = row * row_size
        for block in range(dims // 32):
            off = base + block * 12
            d = np.frombuffer(raw[off:off + 2], dtype=np.float16)[0].astype(np.float32)
            m = np.frombuffer(raw[off + 2:off + 4], dtype=np.float16)[0].astype(np.float32)
            qs = raw[off + 4:off + 12]
            means[row, block, 0] = m
            for j in range(32):
                q = (qs[j // 4] >> (2 * (j % 4))) & 0x03
                decoded_blocks[row, block, j] = d * Q2_CENTROIDS[q]

    if not owht:
        return (decoded_blocks + means).reshape(rows, dims)

    group_size = 128 if dims >= 128 else 32
    groups_per_row = dims // group_size
    blocks_per_group = group_size // 32
    groups = decoded_blocks.reshape(rows, groups_per_row, blocks_per_group, 32).reshape(rows, groups_per_row, group_size)
    if not no_hadamard:
        groups = hadamard_ortho(groups)
    group_means = means.reshape(rows, groups_per_row, blocks_per_group, 1)[:, :, :1, :].reshape(rows, groups_per_row, 1)
    return (groups + group_means).reshape(rows, dims)


def discover_layers(dump_dir: Path) -> list[tuple[str, int]]:
    found = []
    for meta_path in dump_dir.glob("cache_[kv]_set_rows-*_src0.0.meta.txt"):
        match = re.match(r"cache_([kv])_set_rows-(\d+)_src0\.0\.meta\.txt", meta_path.name)
        if match:
            found.append((match.group(1), int(match.group(2))))
    return sorted(found, key=lambda x: (x[1], x[0]))


def compare_one(dump_dir: Path, cache_kind: str, layer: int, *, owht: bool, no_hadamard: bool, clip_ratio: float) -> dict[str, float | int | str]:
    src_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}_src0.0.bin"
    src_meta = parse_meta(src_path.with_suffix(".meta.txt"))
    src_ne = meta_ne(src_meta)
    src = load_f32(src_path, src_ne)

    idx_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}_src1.0.bin"
    idx_meta = parse_meta(idx_path.with_suffix(".meta.txt"))
    indices = load_indices(idx_path, idx_meta)

    cache_path = dump_dir / f"cache_{cache_kind}_set_rows-{layer}.0.bin"
    cache_meta = parse_meta(cache_path.with_suffix(".meta.txt"))
    cache_ne = meta_ne(cache_meta)
    tokens = src.shape[0]
    dims = src.shape[1]

    expected = q2_quantize_dequant(src, owht=owht, no_hadamard=no_hadamard, clip_ratio=clip_ratio)
    cache_rows = dequant_q2_cache(cache_path, cache_ne[1], dims, owht=owht, no_hadamard=no_hadamard)
    actual = cache_rows[indices[:tokens]]

    return {
        "kind": cache_kind.upper(),
        "layer": layer,
        "tokens": tokens,
        "dims": dims,
        "src_ne": "x".join(str(x) for x in src_ne),
        "cache_ne": "x".join(str(x) for x in cache_ne),
        "idx_min": int(indices.min()),
        "idx_max": int(indices.max()),
        "owht": int(owht),
        "no_hadamard": int(no_hadamard),
        "clip_ratio": clip_ratio,
        "nmse_runtime_vs_python": nmse_np(expected, actual),
        "max_abs_runtime_vs_python": float(np.max(np.abs(actual - expected))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze q2 runtime cache set_rows dumps against Python q2_0 writer semantics.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "runs/q2_runtime_cache_dump_current/on/dump")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--owht", action="store_true")
    parser.add_argument("--no-hadamard", action="store_true")
    parser.add_argument("--clip-ratio", type=float, default=0.0)
    parser.add_argument("--clip-ratio-k", type=float, default=None)
    parser.add_argument("--clip-ratio-v", type=float, default=None)
    parser.add_argument("--max-layers", type=int, default=40)
    args = parser.parse_args()

    out_dir = args.out_dir or args.dump_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cache_kind, layer in discover_layers(args.dump_dir):
        if layer >= args.max_layers:
            continue
        clip_ratio = args.clip_ratio
        if cache_kind == "k" and args.clip_ratio_k is not None:
            clip_ratio = args.clip_ratio_k
        if cache_kind == "v" and args.clip_ratio_v is not None:
            clip_ratio = args.clip_ratio_v
        rows.append(compare_one(args.dump_dir, cache_kind, layer, owht=args.owht, no_hadamard=args.no_hadamard, clip_ratio=clip_ratio))

    if not rows:
        raise SystemExit(f"no cache set_rows source dumps found in {args.dump_dir}")

    csv_path = out_dir / "cache_dequant_check.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    max_nmse = max(float(row["nmse_runtime_vs_python"]) for row in rows)
    max_abs = max(float(row["max_abs_runtime_vs_python"]) for row in rows)
    summary = {
        "rows": len(rows),
        "owht": int(args.owht),
        "no_hadamard": int(args.no_hadamard),
        "clip_ratio": args.clip_ratio,
        "clip_ratio_k": args.clip_ratio_k if args.clip_ratio_k is not None else args.clip_ratio,
        "clip_ratio_v": args.clip_ratio_v if args.clip_ratio_v is not None else args.clip_ratio,
        "max_nmse_runtime_vs_python": max_nmse,
        "max_abs_runtime_vs_python": max_abs,
    }
    summary_path = out_dir / "cache_dequant_check_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    md = [
        "| rows | OWHT | no-Hadamard | clip | max runtime-vs-python NMSE | max abs diff |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {summary['rows']} | {summary['owht']} | {summary['no_hadamard']} | {summary['clip_ratio']} | {max_nmse:.8g} | {max_abs:.8g} |",
    ]
    (out_dir / "cache_dequant_check_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
