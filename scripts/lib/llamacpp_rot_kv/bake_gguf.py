from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.lib.llamacpp_rot_kv.errors import MissingDependencyError
from scripts.lib.llamacpp_rot_kv.llama_paths import ensure_gguf_py_on_path

DEFAULT_K_ROT = "k_rotation_qqt_r_h_pbr.pt"
DEFAULT_V_ROT = "v_rotation_sst_r_h_pbr.pt"
ROT_TENSOR_SUFFIXES = (".attn_k_rot.weight", ".attn_v_rot.weight")


def import_deps():
    gguf_py_root = ensure_gguf_py_on_path()
    if str(gguf_py_root) not in sys.path:
        sys.path.insert(0, str(gguf_py_root))
    try:
        import torch
        from gguf import GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType
    except ImportError as exc:
        raise MissingDependencyError(
            "torch and bundled gguf-py are required for OSCAR GGUF baking. "
            f"Install torch and ensure {gguf_py_root} is available: {exc}"
        ) from exc
    return torch, GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType


def field_contents(field: Any) -> Any:
    contents = field.contents
    return contents() if callable(contents) else contents


def derive_output_path(base: Path, name: str | None = None) -> Path:
    if name:
        return base.parent / f"{name}-rot-kv.gguf"
    if base.suffix.lower() == ".gguf":
        return base.with_name(f"{base.stem}-rot-kv{base.suffix}")
    return base.with_name(f"{base.name}-rot-kv.gguf")


def layer_entry(layers: dict, layer_id: int) -> dict:
    if layer_id in layers:
        return layers[layer_id]
    key = str(layer_id)
    if key in layers:
        return layers[key]
    raise KeyError(f"rotation checkpoint is missing layer {layer_id}")


def torch_load(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_rotation(torch, path: Path, *, max_orthogonality_error: float) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing rotation checkpoint: {path}")

    state = torch_load(torch, path)
    layers = state.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path} does not contain a non-empty 'layers' dict")

    out: dict[int, np.ndarray] = {}
    eye_cache: dict[int, Any] = {}
    for layer_id in range(len(layers)):
        entry = layer_entry(layers, layer_id)
        if "rotation" not in entry:
            raise ValueError(f"{path} layer {layer_id} does not contain 'rotation'")

        rot = entry["rotation"].float().cpu()
        if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
            raise ValueError(f"{path} layer {layer_id} rotation must be square, got {tuple(rot.shape)}")

        dim = int(rot.shape[0])
        eye = eye_cache.setdefault(dim, torch.eye(dim, dtype=rot.dtype))
        err = float((rot @ rot.T - eye).abs().max().item())
        if err > max_orthogonality_error:
            raise ValueError(
                f"{path} layer {layer_id} orthogonality error {err:.6g} > {max_orthogonality_error:.6g}"
            )

        out[layer_id] = np.ascontiguousarray(rot.numpy().T.astype(np.float32))

    return out


def gguf_block_count(reader) -> int | None:
    for name, field in reader.fields.items():
        if name.endswith(".block_count"):
            value = field_contents(field)
            if isinstance(value, (list, tuple)):
                value = value[0]
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def base_has_rotation_tensors(reader) -> bool:
    return any(tensor.name.endswith(ROT_TENSOR_SUFFIXES) for tensor in reader.tensors)


def copy_key_values(reader, writer, GGUFValueType) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field_contents(field), value_type, sub_type=sub_type)


@dataclass(slots=True)
class BakeGgufConfig:
    base: Path
    rot_dir: Path
    out_path: Path
    k_rotation_filename: str = DEFAULT_K_ROT
    v_rotation_filename: str = DEFAULT_V_ROT
    max_orthogonality_error: float = 1e-4
    allow_layer_mismatch: bool = False
    replace_rotations: bool = True
    overwrite: bool = False
    dry_run: bool = False


def bake_rotations_to_gguf(config: BakeGgufConfig) -> Path:
    torch, GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType = import_deps()

    base = config.base.resolve()
    out_path = config.out_path.resolve()
    rot_dir = config.rot_dir.resolve()
    k_path = rot_dir / config.k_rotation_filename
    v_path = rot_dir / config.v_rotation_filename

    if not base.is_file():
        raise FileNotFoundError(f"missing base GGUF: {base}")
    if out_path.exists() and not config.overwrite:
        raise FileExistsError(f"output already exists, pass overwrite=true to replace: {out_path}")

    reader = GGUFReader(str(base))
    if base_has_rotation_tensors(reader) and not config.replace_rotations:
        raise ValueError(
            "base GGUF already contains attn_k_rot/attn_v_rot tensors; "
            "set replace_rotations=true to rewrite them"
        )

    k_rot = load_rotation(torch, k_path, max_orthogonality_error=config.max_orthogonality_error)
    v_rot = load_rotation(torch, v_path, max_orthogonality_error=config.max_orthogonality_error)
    if len(k_rot) != len(v_rot):
        raise ValueError(f"K/V layer count mismatch: {len(k_rot)} vs {len(v_rot)}")
    for layer_id in range(len(k_rot)):
        if k_rot[layer_id].shape != v_rot[layer_id].shape:
            raise ValueError(
                f"K/V rotation shape mismatch at layer {layer_id}: "
                f"{k_rot[layer_id].shape} vs {v_rot[layer_id].shape}"
            )

    block_count = gguf_block_count(reader)
    if block_count is not None and block_count != len(k_rot) and not config.allow_layer_mismatch:
        raise ValueError(
            f"GGUF block_count={block_count}, rotation layers={len(k_rot)}. "
            "Set allow_layer_mismatch=true only if this architecture has fewer attention layers."
        )

    if config.dry_run:
        print(
            f"would write {out_path}: base_tensors={len(reader.tensors)} "
            f"rotation_layers={len(k_rot)} dim={k_rot[0].shape[0]} replace={config.replace_rotations}"
        )
        return out_path

    arch = field_contents(reader.get_field("general.architecture"))
    writer = GGUFWriter(str(out_path), arch)
    copy_key_values(reader, writer, GGUFValueType)

    copied_tensors = 0
    for tensor in reader.tensors:
        if tensor.name.endswith(ROT_TENSOR_SUFFIXES):
            continue
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
        copied_tensors += 1

    for layer_id in range(len(k_rot)):
        writer.add_tensor_info(
            f"blk.{layer_id}.attn_k_rot.weight",
            k_rot[layer_id].shape,
            k_rot[layer_id].dtype,
            k_rot[layer_id].nbytes,
            GGMLQuantizationType.F32,
        )
        writer.add_tensor_info(
            f"blk.{layer_id}.attn_v_rot.weight",
            v_rot[layer_id].shape,
            v_rot[layer_id].dtype,
            v_rot[layer_id].nbytes,
            GGMLQuantizationType.F32,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        if tensor.name.endswith(ROT_TENSOR_SUFFIXES):
            continue
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for layer_id in range(len(k_rot)):
        writer.write_tensor_data(k_rot[layer_id], tensor_endianess=reader.endianess)
        writer.write_tensor_data(v_rot[layer_id], tensor_endianess=reader.endianess)
    writer.close()

    print(
        f"wrote {out_path}: copied_tensors={copied_tensors} "
        f"rotation_tensors={2 * len(k_rot)} layers={len(k_rot)} dim={k_rot[0].shape[0]}"
    )
    return out_path
