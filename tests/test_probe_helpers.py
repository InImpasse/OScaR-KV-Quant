import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oscar_kv_quant.probe import ProbeStatus, _health_ok, _served_model_name, _tail


class ProbeHelperTest(unittest.TestCase):
    def test_tail_returns_empty_for_missing_file(self) -> None:
        self.assertEqual(_tail(Path("/tmp/does-not-exist-oscar-kv.log")), "")

    def test_tail_reads_last_n_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.log"
            path.write_text("\n".join(f"line-{i}" for i in range(10)), encoding="utf-8")
            self.assertEqual(_tail(path, n=3), "line-7\nline-8\nline-9")

    @mock.patch("urllib.request.urlopen")
    def test_health_ok_true_on_success(self, urlopen: mock.Mock) -> None:
        urlopen.return_value.__enter__.return_value = object()
        self.assertTrue(_health_ok(12345))

    @mock.patch("urllib.request.urlopen", side_effect=OSError("down"))
    def test_health_ok_false_on_error(self, _urlopen: mock.Mock) -> None:
        self.assertFalse(_health_ok(12345))

    def test_probe_status_defaults_are_non_successful_or_unknown(self) -> None:
        status = ProbeStatus()
        self.assertFalse(status.env_ok)
        self.assertFalse(status.sglang_import_ok)
        self.assertFalse(status.flashinfer_import_ok)
        self.assertIsNone(status.dummy_server_ok)
        self.assertIsNone(status.model_server_ok)
        self.assertIsNone(status.int2_server_ok)

    def test_served_model_name_uses_path_name(self) -> None:
        self.assertEqual(_served_model_name("/models/granite-4.0-1b-base"), "granite-4.0-1b-base")

    def test_served_model_name_keeps_dummy(self) -> None:
        self.assertEqual(_served_model_name("dummy"), "dummy")

    def test_served_model_name_replaces_colon(self) -> None:
        self.assertEqual(_served_model_name("runai://model:version"), "model_version")


if __name__ == "__main__":
    unittest.main()
