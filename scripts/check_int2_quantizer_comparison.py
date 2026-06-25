#!/usr/bin/env python3
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/int2_quantizer_comparison_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    script = (ROOT / "scripts/compare_int2_quantizers.py").read_text()
    common = (ROOT / "third_party/OSCAR/ggml/src/ggml-common.h").read_text()
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()

    centroids = [
        float(re.search(rf"#define Q2_0_LM_C{i} \(([- 0-9.]+)f\)", common).group(1).replace(" ", ""))
        for i in range(4)
    ]
    require(str(centroids) in script,
            "offline comparison q2 centroids must match ggml-common.h")

    summary_path = RUN / "summary.csv"
    require(summary_path.is_file(), "missing int2 quantizer comparison summary")
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    require({row["kind"] for row in rows} == {"Kcur", "Vcur"},
            "comparison must contain Kcur and Vcur summaries")
    for row in rows:
        q2 = float(row["q2_nmse_mean"])
        q2_rot = float(row["q2_rot_nmse_mean"])
        asym_rot = float(row["asym_rot_nmse_mean"])
        turbo2_rot = float(row["turbo2_rot_nmse_mean"])
        turbo3_rot = float(row["turbo3_rot_nmse_mean"])
        require(q2_rot < q2, f"{row['kind']} q2 rotation should reduce ordinary NMSE")
        require(q2_rot < asym_rot, f"{row['kind']} q2+rotation NMSE should be below asymmetric+rotation in current dump")
        require(turbo3_rot < q2_rot, f"{row['kind']} rotated Turbo3 should beat rotated q2 ordinary NMSE in current dump")
        require(turbo3_rot < turbo2_rot, f"{row['kind']} rotated Turbo3 should beat rotated Turbo2 ordinary NMSE in current dump")

    require("Offline quantizer comparison" in report,
            "triage report must include offline quantizer comparison")
    require("ordinary NMSE alone does not explain" in report and "q2 generation failure" in report,
            "triage report must record the interpretation of the offline comparison")
    require("Turbo3" in report and "rotated ordinary NMSE" in report,
            "triage report must document the Turbo3 ordinary-NMSE reference result")

    print("int2 quantizer comparison checks passed")


if __name__ == "__main__":
    main()
