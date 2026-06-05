#!/usr/bin/env python3
import argparse
import csv
import struct
from pathlib import Path


VALUE_TYPES = {
    0: ("uint8", 1, "B"),
    1: ("int8", 1, "b"),
    2: ("uint16", 2, "H"),
    3: ("int16", 2, "h"),
    4: ("uint32", 4, "I"),
    5: ("int32", 4, "i"),
    6: ("float32", 4, "f"),
    7: ("bool", 1, "?"),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", 8, "Q"),
    11: ("int64", 8, "q"),
    12: ("float64", 8, "d"),
}

KV_BYTES_PER_ELEM = {
    "f16": 2.0,
    "bf16": 2.0,
    "f32": 4.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q2_0": 12.0 / 32.0,
}


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.off = 0

    def read(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        val = struct.unpack_from("<" + fmt, self.data, self.off)
        self.off += size
        return val[0] if len(val) == 1 else val

    def read_bytes(self, n: int) -> bytes:
        out = self.data[self.off:self.off + n]
        self.off += n
        return out

    def read_string(self) -> str:
        n = self.read("Q")
        return self.read_bytes(n).decode("utf-8")


def read_value(reader: Reader, value_type: int):
    name, size, fmt = VALUE_TYPES[value_type]
    if name == "string":
        return reader.read_string()
    if name == "array":
        subtype = reader.read("I")
        n = reader.read("Q")
        return [read_value(reader, subtype) for _ in range(n)]
    return reader.read(fmt)


def read_metadata(path: Path) -> dict:
    reader = Reader(path.read_bytes())
    magic = reader.read("I")
    if magic != 0x46554747:
        raise ValueError(f"{path}: not a GGUF file")
    _version = reader.read("I")
    tensor_count = reader.read("Q")
    kv_count = reader.read("Q")
    meta = {}
    for _ in range(kv_count):
        key = reader.read_string()
        value_type = reader.read("I")
        meta[key] = read_value(reader, value_type)
    meta["GGUF.tensor_count"] = tensor_count
    return meta


def get_first(meta: dict, keys: list[str], default=None):
    for key in keys:
        if key in meta:
            value = meta[key]
            return value[0] if isinstance(value, list) else value
    return default


def model_dims(meta: dict) -> dict:
    arch = meta["general.architecture"]
    n_layer = get_first(meta, [f"{arch}.block_count", f"{arch}.decoder_block_count"])
    n_embd = get_first(meta, [f"{arch}.embedding_length"])
    n_head = get_first(meta, [f"{arch}.attention.head_count"])
    n_head_kv = get_first(meta, [f"{arch}.attention.head_count_kv"], n_head)
    key_len = get_first(meta, [f"{arch}.attention.key_length"], n_embd // n_head)
    value_len = get_first(meta, [f"{arch}.attention.value_length"], key_len)
    return {
        "arch": arch,
        "n_layer": int(n_layer),
        "n_embd": int(n_embd),
        "n_head": int(n_head),
        "n_head_kv": int(n_head_kv),
        "key_len": int(key_len),
        "value_len": int(value_len),
    }


def normalize_kv(kv_type: str) -> str:
    if kv_type == "q2":
        return "q2_0"
    if kv_type == "q2hp":
        return "q2_0_hp"
    return kv_type


def storage_kv(kv_type: str) -> str:
    if kv_type.startswith("q2_0_owht"):
        return "q2_0"
    return kv_type


def split_kv(kv_type: str) -> tuple[str, str, str]:
    if "/" in kv_type:
        kv_k, kv_v = kv_type.split("/", 1)
        kv_k = normalize_kv(kv_k)
        kv_v = normalize_kv(kv_v)
        return f"k{kv_k}_v{kv_v}", kv_k, kv_v
    kv_norm = normalize_kv(kv_type)
    return kv_norm, kv_norm, kv_norm


def estimate_axis_bytes(dims: dict, n_ctx: int, kv_type: str, axis: str, hp_slots: int = 0) -> int:
    if kv_type == "q2_0_hp":
        return estimate_axis_bytes(dims, n_ctx, "q2_0", axis) + estimate_axis_bytes(dims, hp_slots, "f16", axis)
    bytes_per_elem = KV_BYTES_PER_ELEM[storage_kv(kv_type)]
    head_dim = dims["key_len"] if axis == "k" else dims["value_len"]
    elems_per_token = dims["n_layer"] * dims["n_head_kv"] * head_dim
    return int(elems_per_token * n_ctx * bytes_per_elem)


def estimate_bytes(dims: dict, n_ctx: int, kv_type: str, hp_slots: int = 0) -> int:
    _, kv_k, kv_v = split_kv(kv_type)
    return (
        estimate_axis_bytes(dims, n_ctx, kv_k, "k", hp_slots)
        + estimate_axis_bytes(dims, n_ctx, kv_v, "v", hp_slots)
    )


def mib(n: int) -> float:
    return n / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Estimate theoretical llama.cpp KV cache storage by GGUF metadata.")
    parser.add_argument("--model", action="append", required=True, help="name:path.gguf")
    parser.add_argument("--contexts", default="512,2048,4096")
    parser.add_argument("--kv-types", default="f16,q8_0,q4_0,q2_0")
    parser.add_argument("--hp-sink", type=int, default=0)
    parser.add_argument("--hp-recent", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    kv_types = [x for x in args.kv_types.split(",") if x]
    hp_slots = args.hp_sink + args.hp_recent
    rows = []
    for spec in args.model:
        name, path_s = spec.split(":", 1)
        meta = read_metadata(Path(path_s))
        dims = model_dims(meta)
        for n_ctx in contexts:
            f16_bytes = estimate_bytes(dims, n_ctx, "f16")
            for kv_type in kv_types:
                kv_label, kv_k, kv_v = split_kv(kv_type)
                n_bytes = estimate_bytes(dims, n_ctx, kv_type, hp_slots)
                rows.append({
                    "model": name,
                    "arch": dims["arch"],
                    "n_layer": dims["n_layer"],
                    "n_head": dims["n_head"],
                    "n_head_kv": dims["n_head_kv"],
                    "key_len": dims["key_len"],
                    "value_len": dims["value_len"],
                    "context": n_ctx,
                    "kv_type": kv_label,
                    "kv_type_k": kv_k,
                    "kv_type_v": kv_v,
                    "hp_slots": hp_slots if kv_k == "q2_0_hp" or kv_v == "q2_0_hp" else 0,
                    "kv_mib": f"{mib(n_bytes):.1f}",
                    "saved_mib_vs_f16": f"{mib(f16_bytes - n_bytes):.1f}",
                    "ratio_vs_f16": f"{(n_bytes / f16_bytes):.3f}",
                })

    fields = list(rows[0].keys()) if rows else []
    if args.out:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
