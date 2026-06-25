#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ON = Path("runs/q2_runtime_cache_dump_current/on/cache_reconstruction_error.csv")
ON_CLIP = Path("runs/q2_runtime_cache_dump_current/on_clip/cache_reconstruction_error.csv")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def load(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"missing {path}")
    return list(csv.DictReader(path.open()))


def max_nmse(rows: list[dict[str, str]], kind: str) -> float:
    vals = [float(r["nmse_recon_vs_src"]) for r in rows if r["kind"] == kind]
    require(vals, f"missing {kind} rows")
    return max(vals)


def min_cos(rows: list[dict[str, str]], kind: str) -> float:
    vals = [float(r["cos_recon_vs_src"]) for r in rows if r["kind"] == kind]
    require(vals, f"missing {kind} rows")
    return min(vals)


def main() -> None:
    on = load(ON)
    clip = load(ON_CLIP)

    require(len(on) == 80 and len(clip) == 80, "expected 40 layers x K/V reconstruction rows")
    require(max_nmse(on, "V") > max_nmse(on, "K"), "V q2 reconstruction NMSE should exceed K in no-clip probe")
    require(min_cos(on, "V") < min_cos(on, "K"), "V q2 reconstruction cosine should be worse than K in no-clip probe")
    require(max_nmse(clip, "K") > max_nmse(on, "K"), "split clipping should worsen K reconstruction max NMSE")
    require(max_nmse(clip, "V") > max_nmse(on, "V"), "split clipping should worsen V reconstruction max NMSE")
    require(max_nmse(on, "V") > 0.30, "V q2 reconstruction error should remain high enough to explain top-token drift")

    print("q2 cache reconstruction error checks passed")


if __name__ == "__main__":
    main()
