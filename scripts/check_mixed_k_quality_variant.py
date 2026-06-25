#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mixed_k_quality_eval_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing {path}")
    return list(csv.DictReader(path.open()))


def one(data: list[dict[str, str]], variant: str, dataset: str) -> dict[str, str]:
    matches = [r for r in data if r["variant"] == variant and r["dataset"] == dataset]
    require(len(matches) == 1, f"expected one row for {variant}/{dataset}, got {len(matches)}")
    return matches[0]


def correct(data: list[dict[str, str]], variant: str, dataset: str) -> int:
    return int(one(data, variant, dataset)["correct"])


def main() -> None:
    cli_eval = (ROOT / "scripts/run_gpqa_gsm8k_cli_eval.py").read_text()
    bench = (ROOT / "scripts/bench_32k_llamacpp_kv.sh").read_text()
    printer = (ROOT / "scripts/print_32k_matrix_commands.py").read_text()
    report = (ROOT / "docs/MIXED_K_Q4_V_Q2_QUALITY.md").read_text()

    require('"oscar_kq4_vq2"' in cli_eval and '"cache_k": "q4_0"' in cli_eval and '"cache_v": "q2_0"' in cli_eval,
            "CLI eval must expose oscar_kq4_vq2 as q4_0/q2_0")
    require('"oscar_kq4_vturbo3"' in cli_eval and '"cache_v": "turbo3"' in cli_eval,
            "CLI eval must expose oscar_kq4_vturbo3 as a q4_0/turbo3 candidate")
    require('case_enabled oscar_kq4_vq2' in bench and '"q4_0" "q2_0"' in bench,
            "32k bench harness must expose oscar_kq4_vq2")
    require('case_enabled oscar_kq4_vturbo3' in bench and '"q4_0" "turbo3"' in bench,
            "32k bench harness must expose oscar_kq4_vturbo3")
    require('("oscar_kq4_vq2", "180", {})' in printer,
            "matrix command printer must expose oscar_kq4_vq2 without q2 ACKs")
    require('("oscar_kq4_vturbo3", "240", {})' in printer,
            "matrix command printer must expose oscar_kq4_vturbo3 without q2 ACKs")
    require("not exact OSCAR INT2" in report,
            "mixed-K quality report must not call q4/q2 exact INT2")

    cmake = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/CMakeLists.txt").read_text()
    fattn = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn.cu").read_text()
    for instance in (
        "template-instances/fattn-vec-instance-q4_0-q2_0.cu",
        "template-instances/fattn-vec-instance-q4_0-turbo3_0.cu",
    ):
        require(instance in cmake, f"CMake must compile {instance} in the default CUDA FA build")
        require((ROOT / f"third_party/OSCAR/ggml/src/ggml-cuda/{instance}").is_file(),
                f"missing CUDA FA instance {instance}")
    require("FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q2_0)" in fattn,
            "CUDA FA dispatch must expose D=128 q4_0/q2_0")
    require("FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_0)" in fattn,
            "CUDA FA dispatch must expose D=128 q4_0/turbo3")
    require("mixed_q4_v_lowp" in fattn and "K->type != V->type && !mixed_q4_v_lowp" in fattn,
            "default CUDA FA support check must allow the mixed q4_0 low-bit V candidates")
    fattn_vec = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-vec.cuh").read_text()
    require("mixed_non_turbo_K_turbo_V ? 4 : 2" in fattn_vec,
            "mixed non-turbo-K/turbo-V vector FA should keep the tested cols_per_block=4 policy")
    require("V_is_turbo || V_is_oscar2 ? (nthreads_V_q / 4 < 1 ? 1 : nthreads_V_q / 4)" in fattn_vec,
            "turbo V vector FA should keep the tested 8-lane D=128 policy")

    data = rows(RUN / "summary.csv")
    data10 = rows(ROOT / "runs/mixed_k_turbo3_quality_eval_10_current/summary.csv")
    for variant in ("baseline_bf16", "oscar_int4", "oscar_int2", "oscar_kq4_vq2", "plain_kq4_vq2"):
        for dataset in ("gpqa", "gsm8k"):
            require(int(one(data, variant, dataset)["total"]) == 3, f"{variant}/{dataset} should use 3-case smoke")

    require(correct(data, "oscar_kq4_vq2", "gpqa") >= correct(data, "oscar_int4", "gpqa"),
            "OSCAR q4K/q2V GPQA smoke should reach INT4/BF16 band")
    require(correct(data, "oscar_kq4_vq2", "gsm8k") >= correct(data, "oscar_int4", "gsm8k"),
            "OSCAR q4K/q2V GSM8K smoke should reach INT4/BF16 band")
    require(correct(data, "oscar_kq4_vq2", "gpqa") > correct(data, "oscar_int2", "gpqa"),
            "OSCAR q4K/q2V should improve over exact q2/q2 on GPQA smoke")
    require(correct(data, "oscar_kq4_vq2", "gsm8k") > correct(data, "oscar_int2", "gsm8k"),
            "OSCAR q4K/q2V should improve over exact q2/q2 on GSM8K smoke")
    require(correct(data, "plain_kq4_vq2", "gpqa") == 0 and correct(data, "plain_kq4_vq2", "gsm8k") == 0,
            "plain q4K/q2V should remain separated from rotated OSCAR quality path")

    for dataset in ("gpqa", "gsm8k"):
        require(correct(data10, "oscar_kq4_vturbo3", dataset) >= correct(data10, "baseline_bf16", dataset),
                f"oscar q4K/turbo3V should reach BF16 band on {dataset} 10-case smoke")
        require(correct(data10, "oscar_kq4_vturbo3", dataset) >= correct(data10, "oscar_int4", dataset),
                f"oscar q4K/turbo3V should reach INT4 band on {dataset} 10-case smoke")
        require(correct(data10, "oscar_kq4_vturbo3", dataset) > correct(data10, "oscar_kq4_vq2", dataset),
                f"oscar q4K/turbo3V should improve over q4K/q2V on {dataset}")

    for run, variant, kv in (
        ("runs/oscar_kq4_vq2_32k_current/summary.csv", "oscar_kq4_vq2", "q4_0/q2_0"),
        ("runs/oscar_kq4_vturbo3_32k_current/summary.csv", "oscar_kq4_vturbo3", "q4_0/turbo3"),
    ):
        summary = rows(ROOT / run)
        item = summary[0]
        require(item["variant"] == variant and item["status"] == "failed" and f'{item["cache_k"]}/{item["cache_v"]}' == kv,
                f"{variant} 32k archive should record current timeout status")
        require(float(item["peak_mib"]) < 4300.0,
                f"{variant} should retain the expected low-memory mixed-KV behavior")
        require(item["pp_tps"] == "" and "missing or invalid" in item["reason"],
                f"{variant} should not claim a fake 32k speed result")

    speed_rows = rows(ROOT / "runs/oscar_kq4_vturbo3_8k_vthreads8_test/summary.csv")
    speed = speed_rows[0]
    require(speed["variant"] == "oscar_kq4_vturbo3" and speed["status"] == "ok",
            "q4/turbo3 cols4 vthreads8 8k archive should be a valid run")
    require(float(speed["pp_tps"]) >= 300.0,
            "q4/turbo3 cols4 vthreads8 8k prefill should retain the measured mixed-FA speedup")
    require(float(speed["peak_mib"]) < 3800.0,
            "q4/turbo3 cols4 vthreads8 8k should retain low peak memory")

    speed16 = rows(ROOT / "runs/oscar_kq4_vturbo3_16k_vthreads8_test/summary.csv")[0]
    require(speed16["variant"] == "oscar_kq4_vturbo3" and speed16["status"] == "ok",
            "q4/turbo3 cols4 vthreads8 16k archive should be a valid run")
    require(float(speed16["pp_tps"]) < 250.0,
            "q4/turbo3 cols4 vthreads8 16k should remain documented as not yet a final 32k-speed solution")

    print("mixed K=q4,V=q2/turbo3 quality variant checks passed")


if __name__ == "__main__":
    main()
