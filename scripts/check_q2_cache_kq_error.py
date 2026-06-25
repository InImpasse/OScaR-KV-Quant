#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/q2_cache_kq_error_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    triage = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("q2 cache KQ reconstruction" in triage,
            "triage doc must record q2 cache KQ reconstruction evidence")
    require("direct scalar q2 cache reader" in triage,
            "triage doc must connect cache KQ evidence to the generic q2 reader")

    csv_path = RUN / "cache_kq_error.csv"
    require(csv_path.is_file(), f"missing {csv_path}")
    rows = list(csv.DictReader(csv_path.open()))
    require(len(rows) == 40, f"expected 40 layer rows, got {len(rows)}")
    require({row["owht"] for row in rows} == {"1"}, "expected OWHT staged cache rows")
    require({row["no_hadamard"] for row in rows} == {"1"}, "expected no-Hadamard staged cache rows")

    min_k_cos = min(float(row["k_cos"]) for row in rows)
    max_k_nmse = max(float(row["k_nmse"]) for row in rows)
    min_kq_cos = min(float(row["kq_cos"]) for row in rows)
    max_kq_nmse = max(float(row["kq_nmse"]) for row in rows)
    require(min_k_cos < 0.95 and max_k_nmse > 0.15,
            "q2 cache K rows should show nontrivial reconstruction loss")
    require(min_kq_cos < 0.85 and max_kq_nmse > 0.30,
            "q2 cache KQ should show larger attention-score drift")

    print("q2 cache KQ error checks passed")


if __name__ == "__main__":
    main()
