import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BenchCliDryRunTest(unittest.TestCase):
    def _write_model_config(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "test-model",
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 4,
                    "hidden_size": 256,
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_bench_dry_run_writes_reports_without_transformers_or_gpu(self) -> None:
        model_dir = self._write_model_config()
        with tempfile.TemporaryDirectory() as out:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oscar_kv_quant.bench",
                    "--profile",
                    "granite",
                    "--model-path",
                    str(model_dir),
                    "--preset",
                    "short",
                    "--modes",
                    "bf16,int2",
                    "--dry-run",
                    "--results-dir",
                    out,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertIn("[bench:dry-run] mode=bf16", proc.stdout)
            self.assertIn("[bench:dry-run] mode=int2", proc.stdout)
            self.assertTrue(list(Path(out).glob("bench_granite_*.csv")))
            self.assertTrue(list(Path(out).glob("bench_granite_*.md")))

    def test_bench_oscar_dry_run_reports_missing_rotations(self) -> None:
        model_dir = self._write_model_config()
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as rot:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oscar_kv_quant.bench",
                    "--profile",
                    "granite",
                    "--model-path",
                    str(model_dir),
                    "--modes",
                    "oscar-int2",
                    "--rot-dir",
                    rot,
                    "--dry-run",
                    "--results-dir",
                    out,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertIn("missing rotation file", proc.stdout)


if __name__ == "__main__":
    unittest.main()
