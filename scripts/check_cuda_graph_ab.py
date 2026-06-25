#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def one(rows: list[dict[str, str]], mode: str, opt: str) -> dict[str, str]:
    matches = [r for r in rows if r.get("mode") == mode and r.get("opt") == opt]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for mode={mode} opt={opt}, got {len(matches)}")
    return matches[0]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise AssertionError(f"mode={row.get('mode')} opt={row.get('opt')} missing {key}")
    return float(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check current CUDA graph 512-token A/B conclusion.")
    parser.add_argument("graph_ab_csv", type=Path)
    args = parser.parse_args()

    rows = load(args.graph_ab_csv)
    off = one(rows, "off", "0")
    on = one(rows, "on", "1")

    for row in (off, on):
        require(row["variant"] == "plain_int2", "graph A/B must be plain_int2")
        require(row["status"] == "ok", "graph A/B rows must be ok")
        require(row["prompt"] == "512", "graph A/B must stay at 512-token smoke")
        require(row["kv"] == "q2_0/q2_0", "graph A/B must use q2_0/q2_0")

    off_pp = as_float(off, "pp_tps")
    on_pp = as_float(on, "pp_tps")
    on_pp_delta_pct = as_float(on, "pp_pct_vs_off")
    require(off_pp > 1500.0, "graph-off 512 q2 pp should remain healthy")
    require(on_pp > 1500.0, "graph-on 512 q2 pp should remain healthy")
    require(on_pp <= off_pp, "graph-on row should not be interpreted as a speedup")
    require(on_pp_delta_pct <= 0.0, "graph-on pp delta percent should be non-positive")

    print(f"CUDA graph A/B checks passed: {args.graph_ab_csv}")


if __name__ == "__main__":
    main()
