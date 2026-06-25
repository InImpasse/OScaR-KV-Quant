#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROT_DIR = ROOT / "rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations"
DEFAULT_GGUF = ROOT / "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def import_deps():
    sys.path.insert(0, str(ROOT / "third_party/OSCAR/gguf-py"))
    try:
        import torch
        from gguf import GGUFReader
    except Exception as exc:
        raise SystemExit(f"missing dependency for rotation alignment check: {exc}") from exc
    return torch, GGUFReader


def load_pt_rot(torch, path: Path) -> np.ndarray:
    state = torch.load(path, map_location="cpu")
    layers = state["layers"]
    mats = []
    for il in range(len(layers)):
        layer = layers[il] if il in layers else layers[str(il)]
        mats.append(layer["rotation"].float().cpu().numpy())
    return np.stack(mats)


def load_gguf_rot(GGUFReader, path: Path, kind: str, n_layer: int) -> np.ndarray:
    reader = GGUFReader(str(path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    mats = []
    for il in range(n_layer):
        name = f"blk.{il}.attn_{kind}_rot.weight"
        require(name in tensors, f"missing GGUF rotation tensor: {name}")
        mats.append(np.asarray(tensors[name].data, dtype=np.float32))
    return np.stack(mats)


def check_kind(torch, GGUFReader, rot_dir: Path, gguf_path: Path, kind: str, filename: str, atol: float) -> None:
    pt = load_pt_rot(torch, rot_dir / filename)
    gguf = load_gguf_rot(GGUFReader, gguf_path, kind, pt.shape[0])

    direct = np.abs(pt - gguf)
    transposed = np.abs(np.transpose(pt, (0, 2, 1)) - gguf)

    require(float(transposed.max()) <= atol, (
        f"{kind} rotation mismatch: GGUF should store PT rotation transposed for ggml_mul_mat; "
        f"max_abs={float(transposed.max()):.8g}, atol={atol}"
    ))
    require(float(direct.max()) > atol, (
        f"{kind} rotation unexpectedly matches PT direct layout; expected transposed GGUF storage"
    ))

    ortho = np.matmul(np.transpose(pt, (0, 2, 1)), pt)
    ident = np.eye(pt.shape[-1], dtype=np.float32)[None, :, :]
    ortho_err = float(np.max(np.abs(ortho - ident)))
    require(ortho_err < 1e-5, f"{kind} rotation orthogonality error too high: {ortho_err:.8g}")

    print(
        f"{kind}: layers={pt.shape[0]} shape={pt.shape[1]}x{pt.shape[2]} "
        f"gguf_matches_pt_transpose=max_abs:{float(transposed.max()):.8g} "
        f"direct_max_abs:{float(direct.max()):.8g} orthogonality_max_abs:{ortho_err:.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Granite OSCAR PT rotations match baked GGUF tensors.")
    parser.add_argument("--rot-dir", type=Path, default=DEFAULT_ROT_DIR)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    require(args.rot_dir.is_dir(), f"missing rotation directory: {args.rot_dir}")
    require(args.gguf.is_file(), f"missing GGUF: {args.gguf}")
    torch, GGUFReader = import_deps()

    check_kind(torch, GGUFReader, args.rot_dir, args.gguf, "k", "k_rotation_qqt_r_h_pbr.pt", args.atol)
    check_kind(torch, GGUFReader, args.rot_dir, args.gguf, "v", "v_rotation_sst_r_h_pbr.pt", args.atol)
    print("Granite rotation alignment checks passed")


if __name__ == "__main__":
    main()
