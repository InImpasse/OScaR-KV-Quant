#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/int2_attention_error_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("Offline attention-output comparison" in report,
            "triage report must include offline attention-output comparison")
    require("q2 OSCAR rot mean" in report and "runtime KQ/softmax behavior" in report,
            "triage report must record the OSCAR q2 attention comparison interpretation")

    summary_path = RUN / "summary.csv"
    require(summary_path.is_file(), "missing int2 attention error summary")
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 1, "attention summary should contain one aggregate row")
    row = rows[0]
    require(row["rows"] == "160", "attention comparison should cover 160 layer/prompt rows")
    q2 = float(row["q2_attn_nmse_mean"])
    q2_rot = float(row["q2_rot_attn_nmse_mean"])
    q2_oscar_rot = float(row["q2_oscar_rot_attn_nmse_mean"])
    q2_oscar_konly = float(row["q2_oscar_konly_attn_nmse_mean"])
    q2_oscar_vonly = float(row["q2_oscar_vonly_attn_nmse_mean"])
    asym_rot = float(row["asym_rot_attn_nmse_mean"])
    turbo2_rot = float(row["turbo2_rot_attn_nmse_mean"])
    turbo3_rot = float(row["turbo3_rot_attn_nmse_mean"])
    require(q2_rot < q2, "rotation should reduce q2 attention-output NMSE")
    require(q2_oscar_rot <= q2_rot * 1.05,
            "OSCAR q2 no-Hadamard split-clip attention-output NMSE should stay near or below rotated q2")
    require(q2_rot < asym_rot, "rotated q2 attention-output NMSE should be below rotated asymmetric int2 in current dump")
    require(q2_oscar_konly > q2_oscar_vonly * 5.0,
            "OSCAR q2 K-only attention error should dominate V-only error in current dump")
    require(q2_oscar_konly > q2_oscar_rot,
            "OSCAR q2 K-only attention error should be worse than combined K/V q2 in current dump")
    require(turbo3_rot < q2_rot, "rotated Turbo3 should beat rotated q2 attention-output NMSE in current dump")
    require(turbo2_rot > q2_rot, "rotated Turbo2 should not be treated as the attention-quality replacement in current dump")
    require("Turbo3" in report and "better 3-bit" in report and "attention-output" in report,
            "triage report must document the Turbo3 attention-quality reference result")
    require("K-only" in report and "V-only" in report and "K=q2" in report,
            "triage report must document the K-only/V-only q2 attention split")

    print("int2 attention error checks passed")


if __name__ == "__main__":
    main()
