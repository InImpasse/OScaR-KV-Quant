"""Diagnostics for OSCAR QKV calibration dumps and rotation checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch


def _load_chunks(layer_dir: Path, name: str, max_chunks: int | None) -> torch.Tensor:
    paths = sorted((layer_dir / name).glob("*.pt"), key=lambda p: int(p.stem))
    paths = [p for p in paths if int(p.stem) != 0]
    if max_chunks is not None:
        paths = paths[:max_chunks]
    if not paths:
        raise FileNotFoundError(f"no non-warmup chunks for {layer_dir / name}")
    return torch.cat(
        [torch.load(path, map_location="cpu", weights_only=True).float() for path in paths],
        dim=0,
    )


def _layer_ids(dump_path: Path, max_layers: int | None) -> list[int]:
    ids = sorted(
        int(path.name.split("_", 1)[1])
        for path in dump_path.iterdir()
        if path.is_dir() and path.name.startswith("layer_")
    )
    return ids[:max_layers] if max_layers is not None else ids


def _rotation(state: dict, layer_id: int) -> torch.Tensor:
    layers = state["layers"]
    entry = layers.get(layer_id, layers.get(str(layer_id)))
    if entry is None:
        raise KeyError(f"missing layer {layer_id}")
    return entry["rotation"].float()


def _int2_asym(x: torch.Tensor, group_size: int) -> torch.Tensor:
    original_shape = x.shape
    if original_shape[-1] % group_size:
        raise ValueError(f"last dim {original_shape[-1]} not divisible by group_size={group_size}")
    grouped = x.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
    val_min = grouped.amin(dim=-1, keepdim=True)
    val_max = grouped.amax(dim=-1, keepdim=True)
    scale = (val_max - val_min).clamp(min=1e-8) / 3.0
    zero = -val_min / scale
    q = (grouped / scale + zero).round().clamp(0, 3)
    deq = (q - zero) * scale
    return deq.reshape(original_shape)


def _flat_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    return x.reshape(-1, head_dim)


def _basic_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.float().reshape(-1)
    abs_y = y.abs()
    return {
        "mean_abs": float(abs_y.mean()),
        "rms": float(torch.sqrt((y * y).mean())),
        "max_abs": float(abs_y.max()),
        "p99_abs": float(torch.quantile(abs_y, 0.99)),
        "p999_abs": float(torch.quantile(abs_y, 0.999)),
        "outlier_6sigma_frac": float((abs_y > 6 * y.std().clamp(min=1e-12)).float().mean()),
    }


def _nmse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    err = (x_hat - x).float()
    denom = (x.float() * x.float()).mean().clamp(min=1e-12)
    return float((err * err).mean() / denom)


def _weighted_error(err_flat: torch.Tensor, cov: torch.Tensor) -> float:
    # Trace(E[e e^T] C) = mean(e C e^T). Normalized outside by callers.
    return float(((err_flat @ cov) * err_flat).sum(dim=1).mean())


def _k_cov(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> torch.Tensor:
    n_heads = q.shape[1] if q.ndim >= 3 else q.shape[0]
    kv_heads = k.shape[1] if k.ndim >= 3 else k.shape[0]
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)
    cov = torch.zeros(head_dim, head_dim)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        cov += qg.T @ qg / qg.shape[0]
    cov /= kv_heads
    return (cov + cov.T) / 2


def _v_cov(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    n_heads = q.shape[1] if q.ndim >= 3 else q.shape[0]
    kv_heads = k.shape[1] if k.ndim >= 3 else k.shape[0]
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)
    k_flat = k.reshape(-1, kv_heads, head_dim)
    v_flat = v.reshape(-1, kv_heads, head_dim)
    n_tokens = q_flat.shape[0]
    cov = torch.zeros(head_dim, head_dim)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        kh = k_flat[:, h, :]
        vh = v_flat[:, h, :]
        qtq = qg.T @ qg / qg.shape[0]
        weights = (kh @ qtq * kh).sum(1)
        weights = weights / weights.sum().clamp(min=1e-12) * n_tokens
        vw = vh * weights.unsqueeze(1).sqrt()
        cov += vw.T @ vw / n_tokens
    cov /= kv_heads
    return (cov + cov.T) / 2


def _target_metrics(
    x: torch.Tensor,
    rotation: torch.Tensor,
    cov: torch.Tensor,
    head_dim: int,
    group_size: int,
) -> dict[str, float | dict[str, float]]:
    x_flat = _flat_heads(x, head_dim)
    x_rot = x_flat @ rotation
    x_q = _int2_asym(x_flat, group_size)
    x_rot_q = _int2_asym(x_rot, group_size)
    err_plain = x_q - x_flat
    err_rot = (x_rot_q - x_rot) @ rotation.T

    plain_nmse = _nmse(x_flat, x_q)
    rot_nmse = _nmse(x_flat, x_flat + err_rot)
    cov_norm = float(torch.trace(cov @ (x_flat.T @ x_flat / x_flat.shape[0])).clamp(min=1e-12))
    plain_weighted = _weighted_error(err_plain, cov) / cov_norm
    rot_weighted = _weighted_error(err_rot, cov) / cov_norm
    return {
        "plain_nmse": plain_nmse,
        "rotated_nmse": rot_nmse,
        "nmse_ratio_rotated_vs_plain": rot_nmse / max(plain_nmse, 1e-12),
        "plain_weighted_error": plain_weighted,
        "rotated_weighted_error": rot_weighted,
        "weighted_error_ratio_rotated_vs_plain": rot_weighted / max(plain_weighted, 1e-12),
        "plain_stats": _basic_stats(x_flat),
        "rotated_stats": _basic_stats(x_rot),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", type=Path, required=True)
    parser.add_argument("--rot-dir", type=Path, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    k_state = torch.load(args.rot_dir / "k_rotation_qqt_r_h_pbr.pt", map_location="cpu", weights_only=False)
    v_state = torch.load(args.rot_dir / "v_rotation_sst_r_h_pbr.pt", map_location="cpu", weights_only=False)

    layer_results = []
    for layer_id in _layer_ids(args.dump_path, args.max_layers):
        layer_dir = args.dump_path / f"layer_{layer_id}"
        q = _load_chunks(layer_dir, "q", args.max_chunks)
        k = _load_chunks(layer_dir, "k", args.max_chunks)
        v = _load_chunks(layer_dir, "v", args.max_chunks)
        k_cov = _k_cov(q, k, args.head_dim)
        v_cov = _v_cov(q, k, v, args.head_dim)
        k_metrics = _target_metrics(
            k, _rotation(k_state, layer_id), k_cov, args.head_dim, args.group_size
        )
        v_metrics = _target_metrics(
            v, _rotation(v_state, layer_id), v_cov, args.head_dim, args.group_size
        )
        layer_results.append(
            {
                "layer": layer_id,
                "tokens": int(q.reshape(-1, q.shape[-2], q.shape[-1]).shape[0]) if q.ndim >= 3 else int(q.shape[0]),
                "k": k_metrics,
                "v": v_metrics,
            }
        )
        print(
            f"layer={layer_id:02d} "
            f"k_weighted_ratio={k_metrics['weighted_error_ratio_rotated_vs_plain']:.4f} "
            f"v_weighted_ratio={v_metrics['weighted_error_ratio_rotated_vs_plain']:.4f} "
            f"k_nmse_ratio={k_metrics['nmse_ratio_rotated_vs_plain']:.4f} "
            f"v_nmse_ratio={v_metrics['nmse_ratio_rotated_vs_plain']:.4f}",
            flush=True,
        )

    def avg(path: tuple[str, ...]) -> float:
        vals = []
        for row in layer_results:
            value = row
            for key in path:
                value = value[key]  # type: ignore[index]
            vals.append(float(value))
        return mean(vals) if vals else math.nan

    summary = {
        "num_layers": len(layer_results),
        "avg_k_weighted_error_ratio": avg(("k", "weighted_error_ratio_rotated_vs_plain")),
        "avg_v_weighted_error_ratio": avg(("v", "weighted_error_ratio_rotated_vs_plain")),
        "avg_k_nmse_ratio": avg(("k", "nmse_ratio_rotated_vs_plain")),
        "avg_v_nmse_ratio": avg(("v", "nmse_ratio_rotated_vs_plain")),
        "avg_k_p999_abs_ratio": avg(("k", "rotated_stats", "p999_abs"))
        / max(avg(("k", "plain_stats", "p999_abs")), 1e-12),
        "avg_v_p999_abs_ratio": avg(("v", "rotated_stats", "p999_abs"))
        / max(avg(("v", "plain_stats", "p999_abs")), 1e-12),
    }
    result = {"summary": summary, "layers": layer_results}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
