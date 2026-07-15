from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.llamacpp_rot_kv.compute_rotation import ComputeRotationConfig, compute_rotations, validate_rotation_checkpoint
from scripts.lib.llamacpp_rot_kv.dump_qkv import (
    DUMP_QKV_STORAGE_DTYPE,
    DumpQkvConfig,
    _build_calibrator_cmd,
    _dump_oskv_streaming,
    write_chunk_from_arrays,
)
from scripts.lib.llamacpp_rot_kv.manifest import dump_stage_artifacts_valid, write_manifest
from scripts.lib.llamacpp_rot_kv.oskv import OSKV_FOOTER, OSKV_HEADER_SIZE, OSKV_LAYER_ENTRY_SIZE, OSKV_MAGIC, OSKV_VERSION, load_oskv

QT_ROOT = ROOT.parent / "quantization-tool"
PAPER_ROT = QT_ROOT / "artifacts" / "granite-4-0-1b-base" / "oscar" / "paper-rotations"
PAPER_META = PAPER_ROT / "calibration_meta.json"
BASE_GGUF = QT_ROOT / "artifacts" / "granite-4-0-1b-base" / "gguf" / "granite-4.0-1b-base.gguf"


def _write_oskv_file(path: Path, *, layer_ids: list[int], n_tokens: int = 4, n_heads: int = 2, head_dim: int = 8):
    header = struct.pack("<4sIIII", OSKV_MAGIC, OSKV_VERSION, n_tokens, len(layer_ids), 0) + (b"\x00" * 44)
    entries: list[bytes] = []
    payload = bytearray()
    for layer_id in layer_ids:
        q = np.arange(n_tokens * n_heads * head_dim, dtype=np.float32).reshape(n_tokens, n_heads, head_dim)
        k = q + 1
        v = q + 2
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        q_offset = OSKV_HEADER_SIZE + len(layer_ids) * OSKV_LAYER_ENTRY_SIZE + len(payload)
        k_offset = q_offset + q.nbytes
        v_offset = k_offset + k.nbytes
        entries.append(
            struct.pack(
                "<IHHHHIQQQQQQ",
                layer_id,
                n_heads,
                n_heads,
                n_heads,
                head_dim,
                0,
                q_offset,
                k_offset,
                v_offset,
                q.nbytes,
                k.nbytes,
                v.nbytes,
            )
        )
        payload.extend(q.tobytes())
        payload.extend(k.tobytes())
        payload.extend(v.tobytes())
    path.write_bytes(header + b"".join(entries) + bytes(payload) + OSKV_FOOTER)


def test_storage_dtype_constant():
    assert DUMP_QKV_STORAGE_DTYPE == "bfloat16"


def test_write_chunk_from_arrays_uses_bf16(tmp_path: Path):
    torch = pytest.importorskip("torch")
    out_dump = tmp_path / "qkv"
    out_dump.mkdir()
    layer_arrays = {
        0: {
            "Qcur": np.ones((4, 2, 8), dtype=np.float32),
            "Kcur": np.ones((4, 2, 8), dtype=np.float32) * 2,
            "Vcur": np.ones((4, 2, 8), dtype=np.float32) * 3,
        }
    }
    tokens = write_chunk_from_arrays(torch, 1, "hello", layer_arrays, out_dump)
    assert tokens == 4
    saved_q = torch.load(out_dump / "layer_0" / "q" / "1.pt", map_location="cpu", weights_only=True)
    saved_seq = torch.load(out_dump / "layer_0" / "seq_lens" / "1.pt", map_location="cpu", weights_only=True)
    assert saved_q.dtype == torch.bfloat16
    assert saved_seq.dtype == torch.int32


def test_build_calibrator_cmd_includes_token_budget(tmp_path: Path):
    cmd = _build_calibrator_cmd(
        bin_path="calibrator.exe",
        model=tmp_path / "model.gguf",
        prompt=None,
        ctx=4096,
        predict=1,
        ngl=0,
        flash_attn="off",
        cache_type_k="bf16",
        cache_type_v="bf16",
        dataset=tmp_path / "dataset.jsonl",
        dump_root=tmp_path / "raw",
        dump_format="oskv",
        token_budget=30000,
    )
    assert "--token-budget" in cmd
    assert cmd[cmd.index("--token-budget") + 1] == "30000"


def test_dump_oskv_streaming_allows_token_budget_early_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")

    def fake_run_oskv(**kwargs):
        handler = kwargs["line_handler"]
        dump_root = kwargs["dump_root"]
        for index in (1, 2):
            oskv_path = dump_root / f"prompt_{index:05d}.oskv"
            _write_oskv_file(oskv_path, layer_ids=[0])
            handler(f"oskv_dump_complete: prompt={index} tokens=15000 path={oskv_path}")

    monkeypatch.setattr(
        "scripts.lib.llamacpp_rot_kv.dump_qkv.run_calibrator_oskv_streaming",
        fake_run_oskv,
    )
    config = DumpQkvConfig(
        model=tmp_path / "model.gguf",
        dataset=tmp_path / "dataset.jsonl",
        out_dir=tmp_path / "work",
        options={},
        dump_token_budget=8,
        calib_profile="paper",
        keep_raw_dumps=False,
    )
    out_dump = config.out_dir / "qkv_dumps" / "llamacpp"
    out_dump.mkdir(parents=True)
    dumped_tokens, captured, _capture_s, _convert_s = _dump_oskv_streaming(
        torch=torch,
        calibrator="calibrator.exe",
        config=config,
        prompts=["a", "b", "c"],
        out_dump=out_dump,
        raw_dir=config.out_dir / "raw_oskv",
        threads_per_worker=1,
    )
    assert captured == 2
    assert dumped_tokens == 8


@pytest.mark.skipif(not (PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt").is_file(), reason="paper rotation pt not available")
def test_paper_rotation_checkpoint_schema():
    torch = pytest.importorskip("torch")
    validate_rotation_checkpoint(torch, PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt")
    validate_rotation_checkpoint(torch, PAPER_ROT / "v_rotation_sst_r_h_pbr.pt")


@pytest.mark.skipif(not (PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt").is_file(), reason="paper rotation pt not available")
def test_paper_rotation_matches_itself():
    torch = pytest.importorskip("torch")
    qt_k = torch.load(PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt", map_location="cpu", weights_only=False)
    assert len(qt_k["layers"]) == 40


@pytest.mark.skipif(not (PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt").is_file(), reason="paper rotation pt not available")
def test_export_smoke_from_paper_rotation(tmp_path: Path):
    from scripts.lib.llamacpp_rot_kv.bake_gguf import BakeGgufConfig, bake_rotations_to_gguf

    torch = pytest.importorskip("torch")
    out_gguf = tmp_path / "granite-rot-kv.gguf"
    baked = bake_rotations_to_gguf(
        BakeGgufConfig(
            base=BASE_GGUF,
            rot_dir=PAPER_ROT,
            out_path=out_gguf,
            overwrite=True,
            replace_rotations=True,
        )
    )
    assert baked.is_file()
    assert baked.stat().st_size > 1_000_000_000
    validate_rotation_checkpoint(torch, PAPER_ROT / "k_rotation_qqt_r_h_pbr.pt")
    validate_rotation_checkpoint(torch, PAPER_ROT / "v_rotation_sst_r_h_pbr.pt")
