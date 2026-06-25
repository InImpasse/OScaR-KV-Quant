#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/bench_32k_llamacpp_kv.sh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_harness(env: dict[str, str]) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        [str(HARNESS)],
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="q2_ramp_gate_guard_") as tmp:
        tmpdir = Path(tmp)
        fake_bench = tmpdir / "llama-bench"
        fake_bench.write_text("#!/usr/bin/env bash\necho fake llama-bench should not run >&2\nexit 99\n")
        fake_bench.chmod(0o755)
        base_model = tmpdir / "base.gguf"
        oscar_model = tmpdir / "oscar.gguf"
        base_model.write_bytes(b"")
        oscar_model.write_bytes(b"")
        fake_bin = tmpdir / "bin"
        fake_bin.mkdir()
        fake_nvidia_smi = fake_bin / "nvidia-smi"
        fake_nvidia_smi.write_text("#!/usr/bin/env bash\necho nvidia-smi should not run >&2\nexit 98\n")
        fake_nvidia_smi.chmod(0o755)

        common = {
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "LLAMA_BENCH": str(fake_bench),
            "BASE_MODEL": str(base_model),
            "OSCAR_MODEL": str(oscar_model),
            "DRY_RUN": "0",
            "RUN_PREFLIGHT": "0",
            "CASES": "oscar_int2",
            "PROMPT_TOKENS": "32768",
            "GEN_TOKENS": "1",
            "REPETITIONS": "1",
            "ACK_HEAVY_32K": "1",
            "ACK_Q2_32K_NOGO": "1",
            "OUT_DIR": str(tmpdir / "out"),
        }

        blocked = run_harness(common)
        require(blocked.returncode == 2, f"expected ramp gate refusal, got {blocked.returncode}")
        require("hold_32k_q2" in blocked.stderr, "refusal should name hold_32k_q2")
        require("ACK_Q2_RAMP_GATE_HOLD=1" in blocked.stderr, "refusal should name ramp gate hold ACK")
        require("fake llama-bench should not run" not in blocked.stderr, "harness should refuse before llama-bench")
        require("nvidia-smi should not run" not in blocked.stderr, "harness should refuse before GPU baseline")
        require(not (tmpdir / "out").exists(), "harness should refuse before creating output directory")

        allowed = run_harness({**common, "ACK_Q2_RAMP_GATE_HOLD": "1"})
        require(
            allowed.returncode == 1 and "Could not read current GPU memory/utilization baseline" in allowed.stderr,
            "with ramp gate ACK, harness should progress to the normal GPU baseline guard",
        )
        require("nvidia-smi should not run" in allowed.stderr, "fake nvidia-smi should prove baseline guard was reached")

    print("q2 ramp gate harness guard checks passed")


if __name__ == "__main__":
    main()
