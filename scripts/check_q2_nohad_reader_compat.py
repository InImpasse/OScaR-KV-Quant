#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_q2_runtime_cache_dump import Q2_CENTROIDS, dequant_q2_cache, meta_ne, parse_meta


ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "runs/q2_kq_softmax_dump_current/oscar_kq2_vbf16/tensors"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def dequant_direct_scalar(path: Path, rows: int, dims: int) -> np.ndarray:
    row_size = (dims // 32) * 12
    raw = path.read_bytes()
    require(len(raw) >= rows * row_size, f"{path} too small for {rows} rows x {dims} dims")
    out = np.empty((rows, dims), dtype=np.float32)

    for row in range(rows):
        base = row * row_size
        for block in range(dims // 32):
            off = base + block * 12
            d = np.frombuffer(raw[off:off + 2], dtype=np.float16)[0].astype(np.float32)
            m = np.frombuffer(raw[off + 2:off + 4], dtype=np.float16)[0].astype(np.float32)
            qs = raw[off + 4:off + 12]
            for j in range(32):
                q = (qs[j // 4] >> (2 * (j % 4))) & 0x03
                out[row, block * 32 + j] = d * Q2_CENTROIDS[q] + m
    return out


def main() -> None:
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    status = (ROOT / "docs/KV_CACHE_CURRENT_STATUS.md").read_text()
    common = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/q2_0.cuh").read_text()
    owht = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/q2_0-owht.cuh").read_text()

    require("LLAMA_KV_NO_HADAMARD=1" in report and "direct scalar reader is compatible" in report,
            "triage report must document no-Hadamard q2 reader compatibility")
    require("Hadamard-applied OWHT" in status and "matching inverse-OWHT reader" in status,
            "status doc must distinguish no-Hadamard q2 from Hadamard-applied OWHT")
    require("__half2float(x[ib].d) * q2_0_centroid_cuda(q) + __half2float(x[ib].m)" in common,
            "direct q2 scalar reader must remain m + d*centroid")
    require("if (apply_hadamard)" in owht and "q2_0_hadamard_cuda(tmp, actual_n)" in owht,
            "OWHT helper must keep Hadamard application explicit")

    meta_path = DUMP / "cache_k_set_rows-0.0.meta.txt"
    require(meta_path.is_file(), f"missing q2 cache dump: {meta_path}")
    meta = parse_meta(meta_path)
    ne = meta_ne(meta)
    dims = ne[0]
    rows = ne[1]
    cache_path = meta_path.with_suffix("").with_suffix(".bin")
    direct = dequant_direct_scalar(cache_path, rows, dims)
    nohad = dequant_q2_cache(cache_path, rows, dims, owht=True, no_hadamard=True)
    max_abs = float(np.max(np.abs(direct - nohad)))
    require(max_abs == 0.0, f"no-Hadamard q2 direct scalar reader should match OWHT no-Hadamard decode, max_abs={max_abs}")

    print("q2 no-Hadamard reader compatibility checks passed")


if __name__ == "__main__":
    main()
