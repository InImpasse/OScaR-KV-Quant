from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

OSKV_MAGIC = b"OSKV"
OSKV_FOOTER = b"ENDO"
OSKV_VERSION = 1
OSKV_HEADER_SIZE = 64
OSKV_LAYER_ENTRY_SIZE = 64


class OskvError(ValueError):
    pass


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise OskvError(f"expected {size} bytes, got {len(data)}")
    return data


def parse_oskv_header(data: bytes) -> tuple[int, int, int]:
    if len(data) != OSKV_HEADER_SIZE:
        raise OskvError(f"OSKV header must be {OSKV_HEADER_SIZE} bytes")
    magic, version, n_tokens, n_layers, _flags = struct.unpack_from("<4sIIII", data, 0)
    if magic != OSKV_MAGIC:
        raise OskvError(f"invalid OSKV magic: {magic!r}")
    if version != OSKV_VERSION:
        raise OskvError(f"unsupported OSKV version: {version}")
    return int(n_tokens), int(n_layers), version


def parse_layer_entry(data: bytes) -> dict[str, int]:
    if len(data) != OSKV_LAYER_ENTRY_SIZE:
        raise OskvError(f"OSKV layer entry must be {OSKV_LAYER_ENTRY_SIZE} bytes")
    unpacked = struct.unpack("<IHHHHIQQQQQQ", data)
    (
        layer_id,
        n_heads_q,
        n_heads_k,
        n_heads_v,
        head_dim,
        _reserved,
        q_offset,
        k_offset,
        v_offset,
        q_size,
        k_size,
        v_size,
    ) = unpacked
    return {
        "layer_id": int(layer_id),
        "n_heads_q": int(n_heads_q),
        "n_heads_k": int(n_heads_k),
        "n_heads_v": int(n_heads_v),
        "head_dim": int(head_dim),
        "q_offset": int(q_offset),
        "k_offset": int(k_offset),
        "v_offset": int(v_offset),
        "q_size": int(q_size),
        "k_size": int(k_size),
        "v_size": int(v_size),
    }


def _load_tensor(blob: bytes, offset: int, size: int, n_tokens: int, n_heads: int, head_dim: int) -> np.ndarray:
    raw = blob[offset : offset + size]
    expected = n_tokens * n_heads * head_dim * 4
    if len(raw) != expected:
        raise OskvError(f"tensor size mismatch: expected {expected} bytes, got {len(raw)}")
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape((n_tokens, n_heads, head_dim)).copy()


def load_oskv(path: Path, *, n_tokens: int | None = None) -> dict[int, dict[str, np.ndarray]]:
    with path.open("rb") as handle:
        header = _read_exact(handle, OSKV_HEADER_SIZE)
        file_tokens, n_layers, _version = parse_oskv_header(header)
        if n_tokens is not None and n_tokens != file_tokens:
            raise OskvError(f"{path}: expected {n_tokens} tokens, header has {file_tokens}")
        n_tokens = file_tokens

        entries = [_read_exact(handle, OSKV_LAYER_ENTRY_SIZE) for _ in range(n_layers)]
        payload_offset = OSKV_HEADER_SIZE + n_layers * OSKV_LAYER_ENTRY_SIZE
        handle.seek(0, 2)
        file_size = handle.tell()
        if file_size < payload_offset + len(OSKV_FOOTER):
            raise OskvError(f"{path}: file too small")
        handle.seek(payload_offset)
        payload = _read_exact(handle, file_size - payload_offset - len(OSKV_FOOTER))
        footer = _read_exact(handle, len(OSKV_FOOTER))
        if footer != OSKV_FOOTER:
            raise OskvError(f"{path}: invalid OSKV footer: {footer!r}")

    layers: dict[int, dict[str, np.ndarray]] = {}
    for entry_bytes in entries:
        entry = parse_layer_entry(entry_bytes)
        layer_id = entry["layer_id"]
        head_dim = entry["head_dim"]
        layers[layer_id] = {
            "Qcur": _load_tensor(
                payload,
                entry["q_offset"] - payload_offset,
                entry["q_size"],
                n_tokens,
                entry["n_heads_q"],
                head_dim,
            ),
            "Kcur": _load_tensor(
                payload,
                entry["k_offset"] - payload_offset,
                entry["k_size"],
                n_tokens,
                entry["n_heads_k"],
                head_dim,
            ),
            "Vcur": _load_tensor(
                payload,
                entry["v_offset"] - payload_offset,
                entry["v_size"],
                n_tokens,
                entry["n_heads_v"],
                head_dim,
            ),
        }
    if not layers:
        raise OskvError(f"{path}: no layer tensors found")
    return layers


def oskv_to_layer_paths(path: Path) -> dict[int, dict[str, np.ndarray]]:
    return load_oskv(path)
