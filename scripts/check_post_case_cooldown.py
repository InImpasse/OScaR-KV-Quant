#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/bench_32k_llamacpp_kv.sh"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cooldown_guard_") as tmp_s:
        tmp = Path(tmp_s)
        bin_dir = tmp / "bin"
        out_dir = tmp / "out"
        bin_dir.mkdir()
        base_model = tmp / "base.gguf"
        oscar_model = tmp / "oscar.gguf"
        base_model.write_text("")
        oscar_model.write_text("")

        counter = tmp / "nvidia_smi_count"
        write_executable(
            bin_dir / "nvidia-smi",
            f"""#!/usr/bin/env bash
count_file={counter}
count=0
if [[ -f "$count_file" ]]; then
  count="$(cat "$count_file")"
fi
count=$((count + 1))
echo "$count" > "$count_file"
if [[ "$*" == *"--query-compute-apps"* ]]; then
  exit 0
fi
if [[ "$*" == *"memory.used,utilization.gpu"* ]]; then
  if (( count == 1 )); then
    echo "100, 0"
  else
    echo "4096, 99"
  fi
else
  echo "100"
fi
""",
        )
        write_executable(bin_dir / "fake-llama-bench", "#!/usr/bin/env bash\nprintf '[]\\n'\n")

        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LLAMA_BENCH": str(bin_dir / "fake-llama-bench"),
            "BASE_MODEL": str(base_model),
            "OSCAR_MODEL": str(oscar_model),
            "OUT_DIR": str(out_dir),
            "CASES": "baseline_bf16",
            "PROMPT_TOKENS": "512",
            "GEN_TOKENS": "1",
            "REPETITIONS": "1",
            "DRY_RUN": "0",
            "RUN_PREFLIGHT": "0",
            "CASE_TIMEOUT_SEC": "5",
            "POST_CASE_COOLDOWN_SEC": "1",
            "POST_CASE_COOLDOWN_POLL_SEC": "0.1",
            "MAX_BASELINE_MIB": "1024",
            "MAX_GPU_UTIL": "10",
        }
        result = subprocess.run(
            [str(HARNESS)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        require(result.returncode == 125, f"expected cooldown failure exit 125, got {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}")
        require("GPU did not cool down" in result.stderr, "cooldown failure should be reported on stderr")
        require((out_dir / "baseline_bf16_p512_n1.summary.txt").exists(), "case summary should still be written")
        require((out_dir / "baseline_bf16_p512_n1.case.txt").exists(), "case metadata should still be written")

    print("post-case cooldown guard checks passed")


if __name__ == "__main__":
    main()
