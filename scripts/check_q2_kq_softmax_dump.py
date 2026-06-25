#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/q2_kq_softmax_dump_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def find(rows: list[dict[str, str]], variant: str, tensor: str, pass_name: str) -> dict[str, str]:
    matches = [row for row in rows if row["variant"] == variant and row["tensor"] == tensor and row["pass"] == pass_name]
    require(len(matches) == 40, f"expected 40 layer rows for {variant}/{tensor}/{pass_name}, got {len(matches)}")
    return min(matches, key=lambda row: float(row["cos_vs_bf16"]))


def main() -> None:
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("No-cache KQ / softmax sanity dump" in report,
            "triage report must document no-cache KQ / softmax sanity dump")
    require("not final evidence about" in report and "KV-cache q2 KQ" in report,
            "triage report must not overstate llama-debug no-cache KQ as cache-KQ evidence")
    require("cache writer still matches the Python q2 writer" in report,
            "triage report must preserve q2 cache writer interpretation for this probe")

    csv_path = RUN / "kq_softmax_drift.csv"
    require(csv_path.is_file(), f"missing {csv_path}")
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    q2_kq = find(rows, "oscar_kq2_vbf16", "kq", "prefill")
    q4_kq = find(rows, "oscar_kq4_vbf16", "kq", "prefill")
    q2_soft = find(rows, "oscar_kq2_vbf16", "kq_soft_max", "prefill")
    q4_soft = find(rows, "oscar_kq4_vbf16", "kq_soft_max", "prefill")
    q2_out = find(rows, "oscar_kq2_vbf16", "kqv_out", "prefill")
    q4_out = find(rows, "oscar_kq4_vbf16", "kqv_out", "prefill")

    require(float(q2_kq["cos_vs_bf16"]) < 0.75, "q2 no-cache raw KQ drift should remain visible in current probe")
    require(float(q4_kq["cos_vs_bf16"]) > 0.98, "q4 no-cache raw KQ drift should remain small in current probe")
    require(float(q2_soft["cos_vs_bf16"]) < float(q4_soft["cos_vs_bf16"]),
            "q2 no-cache softmax drift should be worse than q4")
    require(float(q2_out["cos_vs_bf16"]) < float(q4_out["cos_vs_bf16"]),
            "q2 no-cache kqv_out drift should remain worse than q4 in this probe")
    require(float(q4_out["cos_vs_bf16"]) > 0.90, "q4 no-cache kqv_out should remain coherent in this probe")

    recon = RUN / "kq_reconstruction.csv"
    require(recon.is_file(), f"missing {recon}")
    with recon.open(newline="") as f:
        recon_rows = list(csv.DictReader(f))
    bf16_min = min(float(r["best_cos"]) for r in recon_rows if r["variant"] == "oscar_bf16")
    q4_min = min(float(r["best_cos"]) for r in recon_rows if r["variant"] == "oscar_kq4_vbf16")
    q2_min = min(float(r["best_cos"]) for r in recon_rows if r["variant"] == "oscar_kq2_vbf16")
    require(bf16_min > 0.999, "BF16 no-cache kq should reconstruct from Qcur/Kcur")
    require(q4_min > 0.98, "q4 no-cache kq should reconstruct from Qcur/Kcur")
    require(q2_min < 0.90, "q2 no-cache kq should remain the exceptional low-reconstruction case")

    for variant in ("oscar_bf16", "oscar_kq4_vbf16", "oscar_kq2_vbf16"):
        tensor_dir = RUN / variant / "tensors"
        require(tensor_dir.is_dir(), f"missing tensor dump dir for {variant}")
        require(len(list(tensor_dir.glob("*.meta.txt"))) >= 300, f"missing expected tensor metadata for {variant}")

    print("q2 KQ/softmax dump checks passed")


if __name__ == "__main__":
    main()
