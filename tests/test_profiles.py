import json
import tempfile
import unittest
from pathlib import Path

from oscar_kv_quant.profiles import ModelProfile, resolve_kv_geometry


class ProfileGeometryTest(unittest.TestCase):
    def _write_config(self, config: dict) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return root

    def test_top_level_granite_like_config(self) -> None:
        root = self._write_config(
            {
                "model_type": "granitemoehybrid",
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "num_attention_heads": 8,
                "hidden_size": 1024,
                "layer_types": ["attention", "mamba", "attention", "attention"],
            }
        )
        geometry = resolve_kv_geometry(root, profile=None)
        self.assertEqual(geometry.num_layers, 4)
        self.assertEqual(geometry.num_attention_layers, 3)
        self.assertEqual(geometry.layers_for_kv_estimate, 3)
        self.assertEqual(geometry.num_kv_heads, 2)
        self.assertEqual(geometry.head_dim, 128)
        self.assertEqual(geometry.model_type, "granitemoehybrid")

    def test_nested_gemma_text_config(self) -> None:
        root = self._write_config(
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "num_hidden_layers": 6,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 16,
                    "hidden_size": 2048,
                    "head_dim": 128,
                },
            }
        )
        geometry = resolve_kv_geometry(root, profile=None)
        self.assertEqual(geometry.num_layers, 6)
        self.assertIsNone(geometry.num_attention_layers)
        self.assertEqual(geometry.layers_for_kv_estimate, 6)
        self.assertEqual(geometry.num_kv_heads, 4)
        self.assertEqual(geometry.head_dim, 128)
        self.assertEqual(geometry.model_type, "gemma4_text")

    def test_profile_fallbacks_apply_when_config_omits_geometry(self) -> None:
        root = self._write_config({})
        profile = ModelProfile(
            name="fallback", default_model_path=root, num_layers=7, num_kv_heads=3, head_dim=64
        )
        geometry = resolve_kv_geometry(root, profile=profile)
        self.assertEqual(geometry.num_layers, 7)
        self.assertEqual(geometry.num_kv_heads, 3)
        self.assertEqual(geometry.head_dim, 64)


if __name__ == "__main__":
    unittest.main()
