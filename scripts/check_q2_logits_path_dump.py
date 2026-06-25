#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("runs/q2_logits_path_dump_current")
SUMMARY = ROOT / "summary.csv"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def row(rows: list[dict[str, str]], variant: str, tensor: str) -> dict[str, str]:
    for r in rows:
        if r["variant"] == variant and r["tensor"] == tensor:
            return r
    raise SystemExit(f"missing {variant}/{tensor} row in {SUMMARY}")


def f(r: dict[str, str], key: str) -> float:
    return float(r[key])


def main() -> None:
    require(SUMMARY.exists(), f"missing {SUMMARY}")
    rows = list(csv.DictReader(SUMMARY.open()))

    for variant in ("baseline_bf16", "oscar_bf16", "oscar_int4", "oscar_int2", "oscar_kq2_vbf16", "oscar_kbf16_vq2"):
        for tensor in ("result_norm", "result_output", "__fattn__-39", "kqv_out-39"):
            r = row(rows, variant, tensor)
            require(r["finite"] == "true", f"{variant}/{tensor} must be finite")

    baseline = row(rows, "baseline_bf16", "result_output")
    int4 = row(rows, "oscar_int4", "result_output")
    int2 = row(rows, "oscar_int2", "result_output")
    kq2 = row(rows, "oscar_kq2_vbf16", "result_output")
    vq2 = row(rows, "oscar_kbf16_vq2", "result_output")

    require(f(baseline, "cos_vs_oscar_bf16") > 0.999, "base vs rotated BF16 logits should match closely")
    require(f(int4, "cos_vs_oscar_bf16") > 0.98, "OSCAR INT4 logits should remain close to BF16")
    require(f(int4, "top10_overlap_vs_oscar_bf16") >= 0.7, "OSCAR INT4 top logits should mostly match BF16")

    require(f(int2, "cos_vs_oscar_bf16") > 0.80, "OSCAR INT2 logits should retain the no-Hadamard graph-gate improvement")
    require(f(int2, "cos_vs_oscar_bf16") < f(int4, "cos_vs_oscar_bf16") - 0.05,
            "OSCAR INT2 should still be flagged as materially worse than INT4")
    require(f(int2, "top10_overlap_vs_oscar_bf16") < f(int4, "top10_overlap_vs_oscar_bf16"),
            "OSCAR INT2 top logits should still be less stable than INT4")
    require(f(vq2, "cos_vs_oscar_bf16") > f(kq2, "cos_vs_oscar_bf16"),
            "after the no-Hadamard graph gate, V=q2 drift should no longer dominate K=q2 drift")
    require(f(vq2, "nmse_vs_oscar_bf16") < f(kq2, "nmse_vs_oscar_bf16"),
            "after the no-Hadamard graph gate, V=q2 NMSE should be below K=q2 NMSE")

    print("q2 logits path dump checks passed")


if __name__ == "__main__":
    main()
