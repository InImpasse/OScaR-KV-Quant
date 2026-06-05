import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oscar_kv_quant.bench import (
    MODE_KV,
    _default_prefill_backend,
    _mode_selected_kv_gib,
    _validate_rotation_files,
)
from oscar_kv_quant.kv_estimate import kv_bytes_bf16
from oscar_kv_quant.profiles import KVGeometry


class BenchHelperTest(unittest.TestCase):
    def test_user_facing_int_aliases_map_to_sglang_float_kv_dtypes(self) -> None:
        self.assertEqual(MODE_KV["int8"], "fp8_e4m3")
        self.assertEqual(MODE_KV["int4"], "fp4_e2m1")
        self.assertEqual(MODE_KV["oscar-int2"], "int2")

    @mock.patch("oscar_kv_quant.bench._is_cuda_sm120", return_value=False)
    def test_int2_prefill_defaults_to_fa3_off_sm120(self, _is_sm120: mock.Mock) -> None:
        self.assertEqual(_default_prefill_backend("int2"), "fa3")
        self.assertEqual(_default_prefill_backend("oscar-int2"), "fa3")

    @mock.patch("oscar_kv_quant.bench._is_cuda_sm120", return_value=True)
    def test_int2_prefill_defaults_to_triton_on_sm120(self, _is_sm120: mock.Mock) -> None:
        self.assertEqual(_default_prefill_backend("int2"), "triton")
        self.assertEqual(_default_prefill_backend("oscar-int2"), "triton")

    def test_validate_rotation_files_reports_missing_rot_dir(self) -> None:
        ok, msg = _validate_rotation_files(None)
        self.assertFalse(ok)
        self.assertIn("--rot-dir", msg)

    def test_validate_rotation_files_requires_both_k_and_v_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k_rotation_qqt_r_h_pbr.pt").write_bytes(b"placeholder")
            ok, msg = _validate_rotation_files(root)
        self.assertFalse(ok)
        self.assertIn("v_rotation_sst_r_h_pbr.pt", msg)

    def test_validate_rotation_files_accepts_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k_rotation_qqt_r_h_pbr.pt").write_bytes(b"k")
            (root / "v_rotation_sst_r_h_pbr.pt").write_bytes(b"v")
            ok, msg = _validate_rotation_files(root)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_selected_kv_estimate_uses_attention_layers(self) -> None:
        geometry = KVGeometry(
            num_layers=10,
            num_attention_layers=4,
            num_kv_heads=2,
            head_dim=64,
        )
        selected = _mode_selected_kv_gib(
            "bf16", geometry, seq_len=128, prefix_bf16=64, recent_bf16=256
        )
        expected = kv_bytes_bf16(4, 2, 128, 64) / (1024**3)
        self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
