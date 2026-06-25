#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify_llamacpp_32k_kv_no_gpu.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="no_gpu_verifier_") as tmp_s:
        tmp = Path(tmp_s)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        log = tmp / "nvidia_smi_calls.txt"
        fake = bin_dir / "nvidia-smi"
        fake.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {log}\necho \"unexpected nvidia-smi call\" >&2\nexit 97\n")
        fake.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CHECK_GPU_SNAPSHOT": "0",
            "SKIP_NO_GPU_VERIFIER_SELFTEST": "1",
        }
        result = subprocess.run(
            [str(VERIFIER)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        require(result.returncode == 0, f"verifier failed under fake nvidia-smi\nstdout={result.stdout}\nstderr={result.stderr}")
        require(
            "skipped; set CHECK_GPU_SNAPSHOT=1" in result.stdout,
            "verifier should report skipped GPU snapshot by default",
        )
        require(not log.exists() or log.read_text() == "", "default verifier should not call outer nvidia-smi")

    print("no-GPU verifier nvidia-smi selftest passed")


if __name__ == "__main__":
    main()
