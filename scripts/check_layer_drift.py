#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("runs/q2_logits_path_dump_current")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def min_cos(rows: list[dict[str, str]], variant: str, tensor: str) -> float:
    vals = [float(r["cos_vs_ref"]) for r in rows if r["variant"] == variant and r["tensor"] == tensor]
    require(vals, f"missing {variant}/{tensor} layer drift rows")
    return min(vals)


def worst_layer(rows: list[dict[str, str]], variant: str, tensor: str) -> str:
    vals = [r for r in rows if r["variant"] == variant and r["tensor"] == tensor]
    require(vals, f"missing {variant}/{tensor} layer drift rows")
    return min(vals, key=lambda r: float(r["cos_vs_ref"]))["layer"]


def main() -> None:
    csv_path = ROOT / "layer_drift.csv"
    md_path = ROOT / "layer_drift_summary.md"
    require(csv_path.exists(), f"missing {csv_path}")
    require(md_path.exists(), f"missing {md_path}")
    rows = list(csv.DictReader(csv_path.open()))

    int4 = min_cos(rows, "oscar_int4", "__fattn__")
    int2 = min_cos(rows, "oscar_int2", "__fattn__")
    kq2 = min_cos(rows, "oscar_kq2_vbf16", "__fattn__")
    vq2 = min_cos(rows, "oscar_kbf16_vq2", "__fattn__")

    require(int4 > 0.90, "INT4 per-layer attention drift should remain small")
    require(int2 < 0.40, "INT2 should still show large per-layer attention drift")
    require(kq2 < vq2, "K=q2 should show larger per-layer attention drift than V=q2 in this probe")
    require(worst_layer(rows, "oscar_int2", "__fattn__") in {"11", "22", "24", "34"},
            "INT2 worst layer should stay in the observed mid/late-layer drift cluster")

    print("layer drift checks passed")


if __name__ == "__main__":
    main()
