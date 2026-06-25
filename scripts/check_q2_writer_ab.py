#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("runs/q2_writer_ab_current")
EVAL_ROOTS = (
    Path("runs/q2_writer_ab_cli_eval_owht_noclip_current"),
    Path("runs/q2_writer_ab_cli_eval_plain_writer_current"),
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def row(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    for r in rows:
        if r["variant"] == variant and r["tensor"] == "result_output":
            return r
    raise SystemExit(f"missing result_output row for {variant}")


def score(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("task_states"), dict):
        states = data["task_states"]
        return int(states["correct"]), int(states["total"])
    rows = data.get("rows", data)
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    require(isinstance(rows, list), f"unexpected JSON shape in {path}")
    total = 0
    correct = 0
    for item in rows:
        if isinstance(item, dict) and "correct" in item:
            total += 1
            correct += int(bool(item["correct"]))
    require(total > 0, f"no scored rows in {path}")
    return correct, total


def main() -> None:
    summary = ROOT / "summary.csv"
    combined = ROOT / "combined.md"
    require(summary.exists(), f"missing {summary}")
    require(combined.exists(), f"missing {combined}")

    rows = list(csv.DictReader(summary.open()))
    legacy = row(rows, "q2_owht_legacy_hadamard")
    split = row(rows, "q2_owht_split_clip_nohad")
    no_clip = row(rows, "q2_owht_no_clip_nohad")
    plain = row(rows, "q2_plain_writer_nohad")

    require(float(legacy["cos_vs_oscar_bf16"]) < 0.3,
            "legacy graph Hadamard mismatch should remain flagged as bad")
    require(float(no_clip["cos_vs_oscar_bf16"]) > float(split["cos_vs_oscar_bf16"]),
            "OWHT no-clip should beat split clipping in the q2 writer A/B")
    require(float(plain["cos_vs_oscar_bf16"]) > float(split["cos_vs_oscar_bf16"]),
            "plain writer no-Hadamard should beat split clipping in the q2 writer A/B")

    direct = combined.read_text()
    require("q2_owht_no_clip_nohad:  Solution: 2 + 2 =" in direct,
            "direct prompt should record the no-clip partial improvement")
    require("q2_owht_split_clip_nohad:  [end of text]" in direct,
            "direct prompt should record split-clip regression to EOS")

    for root in EVAL_ROOTS:
        require(root.exists(), f"missing {root}")
        for path in sorted((root / "raw").glob("*.json")):
            correct, total = score(path)
            require(total == 3, f"{path} should contain the 3-case q2 writer probe")
            require(correct == 0, f"{path} must keep exact q2 marked incomplete")

    print("q2 writer A/B checks passed")


if __name__ == "__main__":
    main()
