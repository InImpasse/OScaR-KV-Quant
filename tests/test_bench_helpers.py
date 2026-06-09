import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from oscar_kv_quant.bench import (
    MODE_KV,
    _default_max_total_tokens,
    _default_prefill_backend,
    _fp16_weights_unsafe_for_model,
    _int2_group_size_for_head_dim,
    _mode_selected_kv_gib,
    _oscar_env,
    _resolve_recent_bf16_tokens,
    _server_cmd,
    _short_context_oscar_fallback_mode,
    _validate_rotation_files,
)
from oscar_kv_quant.kv_estimate import kv_bytes_bf16
from oscar_kv_quant.profiles import KVGeometry


class BenchHelperTest(unittest.TestCase):
    def test_user_facing_int_aliases_map_to_sglang_kv_dtypes(self) -> None:
        self.assertEqual(MODE_KV["fp16"], "auto")
        self.assertEqual(MODE_KV["int8"], "int8")
        self.assertEqual(MODE_KV["int4"], "int4")
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

    def test_oscar_env_forwards_tuning_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k_rotation_qqt_r_h_pbr.pt").write_bytes(b"k")
            (root / "v_rotation_sst_r_h_pbr.pt").write_bytes(b"v")
            env = _oscar_env(
                root,
                prefix_tokens=64,
                recent_tokens=128,
                hp_prefix_pool_tokens=256,
                max_quant_tokens=32768,
                hp_max_splits=4,
                scale_dtype="bfloat16",
                fused_rotate_clip_quant=True,
            )
        self.assertEqual(env["SGLANG_MIXED_KV_HP_MAX_SPLITS"], "4")
        self.assertEqual(env["SGLANG_MIXED_KV_SCALE_DTYPE"], "bfloat16")
        self.assertEqual(env["SGLANG_MIXED_KV_MAX_QUANT_TOKENS"], "32768")
        self.assertEqual(env["SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT"], "1")

    def test_recent_bf16_auto_prefers_short_output_speed(self) -> None:
        self.assertEqual(_resolve_recent_bf16_tokens("auto", max_new_tokens=64), 64)
        self.assertEqual(_resolve_recent_bf16_tokens("auto", max_new_tokens=128), 256)
        self.assertEqual(_resolve_recent_bf16_tokens("192", max_new_tokens=64), 192)

    def test_oscar_env_uses_optimized_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k_rotation_qqt_r_h_pbr.pt").write_bytes(b"k")
            (root / "v_rotation_sst_r_h_pbr.pt").write_bytes(b"v")
            env = _oscar_env(
                root,
                prefix_tokens=64,
                recent_tokens=256,
                hp_prefix_pool_tokens=None,
            )
        self.assertEqual(env["SGLANG_MIXED_KV_HP_MAX_SPLITS"], "4")
        self.assertEqual(env["SGLANG_MIXED_KV_SCALE_DTYPE"], "bfloat16")
        self.assertEqual(env["SGLANG_MIXED_KV_MAX_QUANT_TOKENS"], "0")
        self.assertEqual(env["SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT"], "1")

    def test_oscar_runtime_estimate_uses_quant_pool_cap(self) -> None:
        geometry = KVGeometry(
            num_layers=40,
            num_attention_layers=40,
            num_kv_heads=4,
            head_dim=128,
        )
        request_only = _mode_selected_kv_gib(
            "oscar-int2",
            geometry,
            seq_len=1024,
            prefix_bf16=64,
            recent_bf16=256,
            pool_tokens=None,
        )
        capped_pool = _mode_selected_kv_gib(
            "oscar-int2",
            geometry,
            seq_len=1024,
            prefix_bf16=64,
            recent_bf16=256,
            pool_tokens=32768,
        )
        self.assertGreater(capped_pool, request_only)

    def test_oscar_runtime_estimate_includes_hp_reserve(self) -> None:
        geometry = KVGeometry(
            num_layers=40,
            num_attention_layers=40,
            num_kv_heads=4,
            head_dim=128,
        )
        one_req = _mode_selected_kv_gib(
            "oscar-int2",
            geometry,
            seq_len=4096,
            prefix_bf16=64,
            recent_bf16=256,
            pool_tokens=4096,
            max_running_requests=1,
            hp_prefix_pool_tokens=None,
        )
        four_reqs = _mode_selected_kv_gib(
            "oscar-int2",
            geometry,
            seq_len=4096,
            prefix_bf16=64,
            recent_bf16=256,
            pool_tokens=4096,
            max_running_requests=4,
            hp_prefix_pool_tokens=None,
        )
        self.assertGreater(four_reqs, one_req)

    def test_server_cmd_cuda_graph_flags(self) -> None:
        cmd_off = _server_cmd(
            "python",
            Path("model"),
            31888,
            32888,
            "bf16",
            "bf16",
            0.88,
            None,
            True,
            None,
            None,
            None,
            1,
            4,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            enable_memory_saver=False,
        )
        self.assertIn("--disable-cuda-graph", cmd_off)
        self.assertIn("--skip-server-warmup", cmd_off)
        cmd_on = _server_cmd(
            "python",
            Path("model"),
            31888,
            32888,
            "bf16",
            "bf16",
            0.88,
            None,
            True,
            None,
            None,
            None,
            1,
            4,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_memory_saver=True,
        )
        self.assertNotIn("--disable-cuda-graph", cmd_on)
        self.assertIn("--enable-memory-saver", cmd_on)

    def test_server_cmd_fp16_uses_auto_kv_with_float16_model_dtype(self) -> None:
        cmd = _server_cmd(
            "python",
            Path("model"),
            31888,
            32888,
            "auto",
            "fp16",
            0.88,
            None,
            True,
            None,
            None,
            None,
            1,
            4,
        )
        self.assertIn("--kv-cache-dtype", cmd)
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "auto")
        self.assertIn("--dtype", cmd)
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "float16")

    def test_granite_fp16_weights_are_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            (model_path / "config.json").write_text(
                json.dumps(
                    {"model_type": "granitemoehybrid", "torch_dtype": "bfloat16"}
                )
            )
            self.assertTrue(_fp16_weights_unsafe_for_model(model_path))
            cmd = _server_cmd(
                "python",
                model_path,
                31888,
                32888,
                "auto",
                "fp16",
                0.88,
                None,
                True,
                None,
                None,
                None,
                1,
                4,
            )
        self.assertNotIn("--dtype", cmd)

    def test_short_context_oscar_fallback_mode(self) -> None:
        self.assertEqual(
            _short_context_oscar_fallback_mode("oscar-int2", 1024, 4096, "int2"),
            "int2",
        )
        self.assertIsNone(
            _short_context_oscar_fallback_mode("oscar-int2", 4096, 4096, "int2")
        )
        self.assertIsNone(
            _short_context_oscar_fallback_mode("bf16", 1024, 4096, "int2")
        )

    def test_default_max_total_tokens_caps_single_request_long_context(self) -> None:
        self.assertEqual(
            _default_max_total_tokens(
                "oscar-int2",
                prefill_tokens=16384,
                max_new_tokens=128,
                max_running_requests=1,
            ),
            17408,
        )
        self.assertEqual(
            _default_max_total_tokens(
                "bf16",
                prefill_tokens=32768,
                max_new_tokens=128,
                max_running_requests=1,
            ),
            33792,
        )

    def test_default_max_total_tokens_skips_multi_request_and_other_modes(self) -> None:
        self.assertIsNone(
            _default_max_total_tokens(
                "oscar-int2",
                prefill_tokens=16384,
                max_new_tokens=128,
                max_running_requests=2,
            )
        )
        self.assertIsNone(
            _default_max_total_tokens(
                "fp8",
                prefill_tokens=16384,
                max_new_tokens=128,
                max_running_requests=1,
            )
        )

    def test_int2_group_size_falls_back_for_small_head_dim(self) -> None:
        self.assertEqual(_int2_group_size_for_head_dim(128), 128)
        self.assertEqual(_int2_group_size_for_head_dim(64), 64)

    def test_disable_oscar_cuda_graph_overrides_default(self) -> None:
        from oscar_kv_quant.bench import _cuda_graph_enabled

        args = SimpleNamespace(enable_cuda_graph=False, disable_oscar_cuda_graph=True)
        self.assertFalse(_cuda_graph_enabled(args, "oscar-int2"))
        args = SimpleNamespace(enable_cuda_graph=False, disable_oscar_cuda_graph=False)
        self.assertTrue(_cuda_graph_enabled(args, "oscar-int2"))


if __name__ == "__main__":
    unittest.main()
