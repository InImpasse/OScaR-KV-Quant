from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.lib.llamacpp_rot_kv.errors import MissingDependencyError


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(f"torch is required to compute rotations: {exc}") from exc
    return torch


def torch_load(torch, path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_hadamard(torch, n: int):
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(torch, n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def bit_reversal_perm(torch, d: int):
    if d < 1 or d & (d - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {d}")
    bits = int(math.log2(d))
    return torch.tensor([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)])


def make_br_perm_matrix(torch, eigenvalues):
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(torch, d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def layer_dirs(dump_path: Path) -> list[Path]:
    dirs = [p for p in dump_path.iterdir() if p.is_dir() and p.name.startswith("layer_")]
    return sorted(dirs, key=lambda p: int(p.name.split("_", 1)[1]))


def load_tensor(torch, layer_dir: Path, name: str, chunk_id: str | int):
    sub_dir = layer_dir / name
    if chunk_id == "all":
        chunk_paths = sorted(sub_dir.glob("*.pt"), key=lambda p: int(p.stem))
        if not chunk_paths:
            raise FileNotFoundError(f"no chunk files in {sub_dir}")
        tensors = [torch_load(torch, p).float().double() for p in chunk_paths]
        return torch.cat(tensors, dim=0)
    path = sub_dir / f"{chunk_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing dumped tensor: {path}")
    return torch_load(torch, path).float().double()


def eigdecomp_from_cov(torch, cov):
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs, eigvals


def load_qkv_tensors(torch, layer_dir: Path, chunk_id: str | int):
    q = load_tensor(torch, layer_dir, "q", chunk_id)
    k = load_tensor(torch, layer_dir, "k", chunk_id)
    v = load_tensor(torch, layer_dir, "v", chunk_id)
    return q, k, v


def compute_qqt_from_tensors(torch, q, k, head_dim: int):
    n_heads = q.shape[1]
    kv_heads = k.shape[1]
    if n_heads % kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} is not divisible by kv_heads={kv_heads}")
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)

    cov = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        cov += qg.T @ qg / qg.shape[0]
    cov /= kv_heads
    return eigdecomp_from_cov(torch, cov)


def compute_sst_from_tensors(torch, q, k, v, head_dim: int):
    n_heads = q.shape[1]
    kv_heads = k.shape[1]
    if n_heads % kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} is not divisible by kv_heads={kv_heads}")
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)
    k_flat = k.reshape(-1, kv_heads, head_dim)
    v_flat = v.reshape(-1, kv_heads, head_dim)
    n_tokens = q_flat.shape[0]

    cov = torch.zeros(head_dim, head_dim, dtype=torch.float64)
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
    return eigdecomp_from_cov(torch, cov)


def compute_qqt(torch, layer_dir: Path, chunk_id: str | int, head_dim: int):
    q, k, _v = load_qkv_tensors(torch, layer_dir, chunk_id)
    return compute_qqt_from_tensors(torch, q, k, head_dim)


def compute_sst(torch, layer_dir: Path, chunk_id: str | int, head_dim: int):
    q, k, v = load_qkv_tensors(torch, layer_dir, chunk_id)
    return compute_sst_from_tensors(torch, q, k, v, head_dim)


def compose_rotation(torch, rotation, eigvals, hadamard, composition: str):
    pbr = make_br_perm_matrix(torch, eigvals)
    if composition == "plain":
        return rotation
    if composition == "pbr":
        return pbr
    if composition == "br":
        return rotation @ pbr
    if composition == "br_h128":
        return rotation @ pbr @ hadamard
    if composition == "r_h":
        return rotation @ hadamard
    if composition == "h_pbr":
        return hadamard @ pbr
    if composition == "h_r_pbr":
        return hadamard @ rotation @ pbr
    if composition == "h_pbr_r":
        return hadamard @ pbr @ rotation
    if composition == "r_h_pbr":
        return rotation @ hadamard @ pbr
    raise ValueError(f"unknown composition: {composition}")


def empty_result(objective: str) -> dict:
    return {"format_version": 1, "objective": objective, "source_grouping": "layer", "layers": {}}


def add_layer(result: dict, layer_id: int, rotation, eigvals) -> None:
    result["layers"][layer_id] = {
        "layer_id": layer_id,
        "rotation": rotation.float().contiguous(),
        "eigenvalues": eigvals.float().contiguous(),
    }


def write_rotation_meta(
    output_dir: Path,
    dump_path: Path,
    calibration_meta: dict,
    method: str,
    composition: str,
    head_dim: int,
    num_layers: int,
    stats: dict,
) -> None:
    meta = {
        "format_version": 1,
        "source": "llama.cpp QKV dump",
        "method": method,
        "composition": composition,
        "recipe": "OSCAR qqt_sst calibrated spectral covariance + Hadamard + bit-reversal",
        "head_dim": head_dim,
        "num_layers": num_layers,
        "dump_path": str(dump_path),
        "calibration": calibration_meta,
        "rotation_files": stats,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output_dir / "rotation_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def rotation_orthogonality_error(torch, rotation) -> float:
    dim = int(rotation.shape[0])
    eye = torch.eye(dim, dtype=rotation.dtype)
    return float((rotation @ rotation.T - eye).abs().max().item())


def validate_rotation_checkpoint(torch, path: Path, *, max_orthogonality_error: float = 1e-4) -> dict[str, float | str]:
    state = torch_load(torch, path)
    layers = state.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path} does not contain a non-empty 'layers' dict")

    max_err = 0.0
    for layer_id, entry in layers.items():
        rot = entry["rotation"].double()
        err = rotation_orthogonality_error(torch, rot)
        if err > max_orthogonality_error:
            raise ValueError(
                f"{path} layer {layer_id} orthogonality error {err:.6g} > {max_orthogonality_error:.6g}"
            )
        max_err = max(max_err, err)
    return {"path": str(path), "max_orthogonality_error": max_err}


@dataclass(slots=True)
class ComputeRotationConfig:
    dump_path: Path
    output_dir: Path
    head_dim: int = 128
    chunk_id: str | int = "all"
    method: str = "qqt_sst"
    composition: str = "r_h_pbr"
    calibration_meta: Path | None = None
    dry_run: bool = False


def compute_rotations(config: ComputeRotationConfig) -> dict[str, Path]:
    torch = import_torch()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = config.dump_path.resolve()
    dirs = layer_dirs(dump_path)
    if not dirs:
        raise FileNotFoundError(f"no layer_* dirs found under {dump_path}")

    calibration_meta: dict = {}
    if config.calibration_meta and config.calibration_meta.is_file():
        calibration_meta = json.loads(config.calibration_meta.read_text(encoding="utf-8"))

    if config.dry_run:
        print(
            f"would compute rotations for {len(dirs)} layers from {dump_path} "
            f"method={config.method} composition={config.composition}"
        )
        return {
            "k": output_dir / f"k_rotation_qqt_{config.composition}.pt",
            "v": output_dir / f"v_rotation_sst_{config.composition}.pt",
        }

    hadamard = build_hadamard(torch, config.head_dim)
    results = {
        ("k", "qqt"): empty_result(f"qqt_{config.composition}"),
        ("v", "sst"): empty_result(f"sst_{config.composition}"),
    }
    stats: dict[str, dict[str, float | str]] = {}
    eye = torch.eye(config.head_dim, dtype=torch.float64)

    print(f"Found {len(dirs)} layers in {dump_path}")
    print(f"Method={config.method} composition={config.composition} chunk={config.chunk_id}")
    for layer_dir in dirs:
        layer_id = int(layer_dir.name.split("_", 1)[1])
        q, k, v = load_qkv_tensors(torch, layer_dir, config.chunk_id)
        k_rot, k_eigvals = compute_qqt_from_tensors(torch, q, k, config.head_dim)
        v_rot, v_eigvals = compute_sst_from_tensors(torch, q, k, v, config.head_dim)
        k_out = compose_rotation(torch, k_rot, k_eigvals, hadamard, config.composition)
        v_out = compose_rotation(torch, v_rot, v_eigvals, hadamard, config.composition)
        k_err = float((k_out @ k_out.T - eye).abs().max().item())
        v_err = float((v_out @ v_out.T - eye).abs().max().item())
        add_layer(results[("k", "qqt")], layer_id, k_out, k_eigvals)
        add_layer(results[("v", "sst")], layer_id, v_out, v_eigvals)
        print(f"  Layer {layer_id:>2}: K(qqt)={k_err:.1e}, V(sst)={v_err:.1e}")

    files = {
        "k": output_dir / f"k_rotation_qqt_{config.composition}.pt",
        "v": output_dir / f"v_rotation_sst_{config.composition}.pt",
    }
    torch.save(results[("k", "qqt")], str(files["k"]))
    torch.save(results[("v", "sst")], str(files["v"]))
    for target, path in files.items():
        stats[target] = validate_rotation_checkpoint(torch, path)

    write_rotation_meta(
        output_dir,
        dump_path,
        calibration_meta,
        config.method,
        config.composition,
        config.head_dim,
        len(dirs),
        stats,
    )
    print(f"Saved: {output_dir / 'rotation_meta.json'}")
    return files
