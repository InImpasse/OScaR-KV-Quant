#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "runs/q2_logits_path_dump_current"
EVAL = ROOT / "runs/gpqa_gsm8k_cli_eval_current/summary.csv"
DIRECT = ROOT / "runs/q2_direct_prompt_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing {path}")
    return list(csv.DictReader(path.open()))


def row(rows: list[dict[str, str]], **want: str) -> dict[str, str]:
    found = [r for r in rows if all(r.get(k) == v for k, v in want.items())]
    require(len(found) == 1, f"expected one row for {want}, got {len(found)}")
    return found[0]


def min_float(rows: list[dict[str, str]], key: str) -> float:
    require(rows, "empty row set")
    return min(float(r[key]) for r in rows)


def max_float(rows: list[dict[str, str]], key: str) -> float:
    require(rows, "empty row set")
    return max(float(r[key]) for r in rows)


def main() -> None:
    triage = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()
    require("INT4 accuracy boundary" in triage,
            "triage doc must include an explicit INT4 accuracy boundary section")
    require("INT4 is a healthy q4 control" in triage,
            "triage doc must state that INT4 is not the same failure class as INT2")

    summary = load_csv(DUMP / "summary.csv")
    top = load_csv(DUMP / "top_token_summary.csv")
    layer = load_csv(DUMP / "layer_drift.csv")
    eval_rows = load_csv(EVAL)

    bf16 = row(summary, variant="baseline_bf16", tensor="result_output")
    int4 = row(summary, variant="oscar_int4", tensor="result_output")
    int2 = row(summary, variant="oscar_int2", tensor="result_output")
    kq2 = row(summary, variant="oscar_kq2_vbf16", tensor="result_output")

    require(float(bf16["cos_vs_oscar_bf16"]) > 0.999,
            "baseline BF16 and rotated BF16 should agree at final logits")
    require(float(int4["cos_vs_oscar_bf16"]) > 0.99,
            "OSCAR INT4 final logits should remain close to BF16")
    require(float(int4["nmse_vs_oscar_bf16"]) < 0.02,
            "OSCAR INT4 final-logit NMSE should remain small")
    require(float(int4["top10_overlap_vs_oscar_bf16"]) >= 0.9,
            "OSCAR INT4 top10 overlap should remain high")
    require(float(int4["cos_vs_oscar_bf16"]) > float(kq2["cos_vs_oscar_bf16"]) + 0.05,
            "OSCAR INT4 should be clearly better than K=q2")
    require(float(int4["cos_vs_oscar_bf16"]) > float(int2["cos_vs_oscar_bf16"]) + 0.10,
            "OSCAR INT4 should be clearly better than exact INT2")

    int4_top = row(top, variant="oscar_int4")
    bf16_top = row(top, variant="oscar_bf16")
    int2_top = row(top, variant="oscar_int2")
    require(int4_top["top_id"] == bf16_top["top_id"] == "220",
            "OSCAR INT4 should preserve the BF16 top token in the direct prompt dump")
    require(int2_top["top_id"] != bf16_top["top_id"],
            "OSCAR INT2 should remain a different top-token failure")

    int4_layers = [r for r in layer if r["variant"] == "oscar_int4" and r["tensor"] == "__fattn__"]
    int2_layers = [r for r in layer if r["variant"] == "oscar_int2" and r["tensor"] == "__fattn__"]
    require(len(int4_layers) == 40 and len(int2_layers) == 40,
            "expected 40 layer drift rows for INT4 and INT2")
    require(min_float(int4_layers, "cos_vs_ref") > 0.94,
            "INT4 per-layer attention cosine should remain high")
    require(max_float(int4_layers, "nmse_vs_ref") < 0.12,
            "INT4 per-layer attention NMSE should remain bounded")
    require(min_float(int2_layers, "cos_vs_ref") < 0.40,
            "INT2 per-layer attention should remain clearly degraded")

    for dataset in ("gpqa", "gsm8k"):
        base_eval = row(eval_rows, variant="baseline_bf16", dataset=dataset)
        oscar_int4_eval = row(eval_rows, variant="oscar_int4", dataset=dataset)
        plain_int4_eval = row(eval_rows, variant="plain_int4", dataset=dataset)
        oscar_int2_eval = row(eval_rows, variant="oscar_int2", dataset=dataset)
        plain_int2_eval = row(eval_rows, variant="plain_int2", dataset=dataset)
        require(int(oscar_int4_eval["total"]) == 50 and int(plain_int4_eval["total"]) == 50,
                f"{dataset} INT4 eval should use the 50-sample run")
        require(int(oscar_int2_eval["total"]) == 3 and int(plain_int2_eval["total"]) == 3,
                f"{dataset} INT2 eval should remain a short failure probe, not comparable as a full score")
        require(abs(float(oscar_int4_eval["accuracy"]) - float(base_eval["accuracy"])) <= 0.05,
                f"{dataset} OSCAR INT4 should remain close to BF16 in the 50-sample smoke")
        require(abs(float(plain_int4_eval["accuracy"]) - float(base_eval["accuracy"])) <= 0.05,
                f"{dataset} plain INT4 should remain close to BF16 in the 50-sample smoke")
        require(float(oscar_int2_eval["accuracy"]) == 0.0 and float(plain_int2_eval["accuracy"]) == 0.0,
                f"{dataset} INT2 short probe should remain failed")

    plain_direct = (DIRECT / "plain_int4.out").read_text(errors="replace")
    oscar_direct = (DIRECT / "oscar_int4.out").read_text(errors="replace")
    plain_int2_direct = (DIRECT / "plain_int2.out").read_text(errors="replace")
    require("Answer: 4" in plain_direct and "Answer: 4" in oscar_direct,
            "plain/oscar INT4 direct prompt should answer 4")
    require("Answer: 4" not in plain_int2_direct,
            "plain INT2 direct prompt should not be mistaken for healthy INT4")

    print("INT4 accuracy boundary checks passed")


if __name__ == "__main__":
    main()
