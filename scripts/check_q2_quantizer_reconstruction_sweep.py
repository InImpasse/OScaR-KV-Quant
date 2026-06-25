#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


CSV = Path("runs/q2_quantizer_reconstruction_sweep_current/q2_quantizer_reconstruction_sweep.csv")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def stats(rows: list[dict[str, str]], kind: str, mode: str) -> tuple[float, float, float]:
    sub = [r for r in rows if r["kind"] == kind and r["mode"] == mode]
    require(len(sub) == 40, f"expected 40 rows for {kind}/{mode}")
    nmse = [float(r["nmse"]) for r in sub]
    cos = [float(r["cosine"]) for r in sub]
    return sum(nmse) / len(nmse), max(nmse), min(cos)


def main() -> None:
    require(CSV.exists(), f"missing {CSV}")
    rows = list(csv.DictReader(CSV.open()))

    for kind in ("K", "V"):
        plain_mean, plain_max, plain_min_cos = stats(rows, kind, "plain_q2")
        owht_mean, owht_max, owht_min_cos = stats(rows, kind, "owht_no_clip")
        clip_mean, clip_max, clip_min_cos = stats(rows, kind, "owht_split_clip")

        require(owht_mean < plain_mean, f"{kind} OWHT no-clip should beat plain q2 mean NMSE")
        require(owht_mean < clip_mean, f"{kind} OWHT no-clip should beat split clip mean NMSE")
        require(owht_max < plain_max, f"{kind} OWHT no-clip should beat plain max NMSE")
        require(owht_max < clip_max, f"{kind} OWHT no-clip should beat split clip max NMSE")
        require(owht_min_cos > plain_min_cos, f"{kind} OWHT no-clip should beat plain min cosine")
        require(owht_min_cos > clip_min_cos, f"{kind} OWHT no-clip should beat split clip min cosine")

    print("q2 quantizer reconstruction sweep checks passed")


if __name__ == "__main__":
    main()
