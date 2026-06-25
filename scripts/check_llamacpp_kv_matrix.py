#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def one(rows: list[dict[str, str]], variant: str, prompt: str) -> dict[str, str]:
    matches = [r for r in rows if r.get("variant") == variant and r.get("prompt") == prompt]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {variant} prompt {prompt}, got {len(matches)}")
    return matches[0]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise AssertionError(f"{row.get('variant')} prompt {row.get('prompt')} missing {key}")
    return float(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check current llama.cpp KV matrix conclusions.")
    parser.add_argument("combined_csv", type=Path)
    args = parser.parse_args()

    rows = load(args.combined_csv)

    bf16 = one(rows, "baseline_bf16", "32768")
    require(bf16["status"] == "ok", "32k BF16 baseline must be ok")
    require(bf16["kv"] == "bf16/bf16", "32k BF16 baseline must use bf16/bf16 KV")
    require(as_float(bf16, "pp_tps") > 2000.0, "32k BF16 pp should remain healthy")

    for variant in ("oscar_int4", "plain_int4"):
        row = one(rows, variant, "32768")
        require(row["status"] == "ok", f"{variant} 32k must be ok")
        require(row["kv"] == "q4_0/q4_0", f"{variant} 32k must use q4_0/q4_0")
        peak_saved = as_float(row, "peak_saved_vs_bf16_mib")
        kv_saved = as_float(row, "kv_saved_vs_bf16_mib")
        require(abs(peak_saved - kv_saved) <= 64.0, f"{variant} memory savings no longer track KV savings")
        require(as_float(row, "pp_tps") > 2000.0, f"{variant} 32k pp should remain healthy")

    turbo2 = one(rows, "turbo2_streamk", "32768")
    require(turbo2["status"] == "ok", "32k turbo2_streamk must be ok")
    require(turbo2["kv"] == "turbo2/turbo2", "32k turbo2_streamk must use turbo2/turbo2 KV")
    require(as_float(turbo2, "pp_tps") > as_float(bf16, "pp_tps"), "32k turbo2_streamk pp should beat BF16 as a plain Turbo2 comparison")
    require(as_float(turbo2, "peak_mib") < as_float(bf16, "peak_mib"), "32k turbo2_streamk peak should be lower than BF16")

    oscar_turbo2 = one(rows, "oscar_turbo2_streamk", "32768")
    require(oscar_turbo2["status"] == "ok", "32k oscar_turbo2_streamk must be ok")
    require(oscar_turbo2["kv"] == "turbo2/turbo2", "32k oscar_turbo2_streamk must use turbo2/turbo2 KV")
    require(as_float(oscar_turbo2, "pp_tps") > as_float(bf16, "pp_tps"), "32k oscar_turbo2_streamk pp must beat BF16")
    require(as_float(oscar_turbo2, "peak_mib") < as_float(bf16, "peak_mib"), "32k oscar_turbo2_streamk peak must be lower than BF16")
    require(abs(as_float(oscar_turbo2, "peak_saved_vs_bf16_mib") - as_float(oscar_turbo2, "kv_saved_vs_bf16_mib")) <= 256.0, "oscar turbo2 memory savings should roughly track KV savings")

    for variant in ("plain_int2", "oscar_int2"):
        row = one(rows, variant, "16384")
        require(row["status"] == "ok", f"{variant} 16k gate must be ok")
        require(row["kv"] == "q2_0/q2_0", f"{variant} 16k must use q2_0/q2_0")
        require(100.0 <= as_float(row, "pp_tps") <= 300.0, f"{variant} 16k pp should reflect slow q2 path")

    q2_32k = one(rows, "oscar_int2", "32768")
    require(q2_32k["status"] == "failed", "32k oscar_int2 should remain marked failed after NO-GO")
    require(q2_32k["kv"] == "q2_0/q2_0", "32k oscar_int2 must use q2_0/q2_0")
    require(q2_32k.get("pp_tps", "") == "", "32k oscar_int2 must not report fake pp")
    require("missing or invalid" in q2_32k.get("reason", ""), "32k oscar_int2 failure reason should mention invalid JSON")

    print(f"matrix checks passed: {args.combined_csv}")


if __name__ == "__main__":
    main()
