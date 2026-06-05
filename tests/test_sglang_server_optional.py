import importlib.util
import os
import subprocess
import sys
import unittest


class OptionalSGLangServerTest(unittest.TestCase):
    def test_dummy_server_probe_when_explicitly_enabled(self) -> None:
        if os.environ.get("OSCAR_KV_RUN_SERVER_TESTS") != "1":
            raise unittest.SkipTest("set OSCAR_KV_RUN_SERVER_TESTS=1 to run server tests")
        if importlib.util.find_spec("sglang") is None:
            raise unittest.SkipTest("sglang is not installed")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "oscar_kv_quant.probe",
                "--try-dummy-server",
                "--timeout-s",
                "45",
                "--port",
                os.environ.get("OSCAR_KV_TEST_PORT", "31991"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn('"dummy_server_ok": true', proc.stdout)


if __name__ == "__main__":
    unittest.main()
