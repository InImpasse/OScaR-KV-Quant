import importlib.util
import subprocess
import sys
import unittest

RUN_SERVER_TESTS = False
TEST_PORT = 31991


class OptionalSGLangServerTest(unittest.TestCase):
    def test_dummy_server_probe_when_explicitly_enabled(self) -> None:
        if not RUN_SERVER_TESTS:
            raise unittest.SkipTest("run with --run-server-tests to enable server tests")
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
                str(TEST_PORT),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn('"dummy_server_ok": true', proc.stdout)


if __name__ == "__main__":
    argv = [sys.argv[0]]
    rest = sys.argv[1:]
    while rest:
        arg = rest.pop(0)
        if arg == "--run-server-tests":
            RUN_SERVER_TESTS = True
        elif arg == "--test-port":
            if not rest:
                raise SystemExit("--test-port requires a value")
            TEST_PORT = int(rest.pop(0))
        else:
            argv.append(arg)
    unittest.main(argv=argv)
