import json
import tempfile
import unittest
from pathlib import Path

from oscar_kv_quant.rotation_gate import main


class RotationGateTest(unittest.TestCase):
    def _write_rotation_meta(self, root: Path, tokens: int = 30000) -> None:
        (root / "rotation_meta.json").write_text(
            json.dumps(
                {
                    "method": "qqt_sst",
                    "composition": "r_h_pbr",
                    "calibration": {
                        "dumped_tokens": tokens,
                        "num_prompts_captured": 64,
                        "runner_error_prompts": 0,
                    },
                    "rotation_files": {
                        "k": {"max_orthogonality_error": 2e-6},
                        "v": {"max_orthogonality_error": 3e-6},
                    },
                }
            )
        )

    def _write_metrics(self, root: Path, score: float) -> None:
        root.mkdir()
        (root / "metrics.json").write_text(json.dumps({"score": score}))

    def test_passes_paper_calibration_and_accuracy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rot_dir = root / "rot"
            rot_dir.mkdir()
            self._write_rotation_meta(rot_dir)
            bf16 = root / "bf16"
            oscar = root / "oscar"
            self._write_metrics(bf16, 0.70)
            self._write_metrics(oscar, 0.67)

            rc = main(
                [
                    "--rot-dir",
                    str(rot_dir),
                    "--bf16-eval-dir",
                    str(bf16),
                    "--oscar-eval-dir",
                    str(oscar),
                    "--max-accuracy-drop",
                    "0.05",
                ]
            )

        self.assertEqual(rc, 0)

    def test_fails_weak_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rot_dir = Path(tmp) / "rot"
            rot_dir.mkdir()
            self._write_rotation_meta(rot_dir, tokens=2000)

            rc = main(["--rot-dir", str(rot_dir)])

        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
