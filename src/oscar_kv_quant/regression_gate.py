"""Validate bench CSV rows against memory/speed regression thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateScenario:
    name: str
    min_memory_ratio: float
    min_steady_vs_int2: float
    min_steady_vs_bf16: float
    min_flush_vs_steady: float | None
    min_request_tok_s_32k: float | None
    min_flush_median_tok_s: float | None
    description: str


GATE_SCENARIOS: dict[str, GateScenario] = {
    "memory": GateScenario(
        name="memory",
        min_memory_ratio=0.35,
        min_steady_vs_int2=0.0,
        min_steady_vs_bf16=0.0,
        min_flush_vs_steady=0.35,
        min_request_tok_s_32k=None,
        min_flush_median_tok_s=None,
        description="Prioritize KV compression; soft decode floors.",
    ),
    "balanced": GateScenario(
        name="balanced",
        min_memory_ratio=0.35,
        min_steady_vs_int2=0.70,
        min_steady_vs_bf16=0.65,
        min_flush_vs_steady=0.05,
        min_request_tok_s_32k=4.0,
        min_flush_median_tok_s=2.5,
        description=(
            "Default product target: oscar trades mixed HP windows for ~5-6x KV "
            "savings and accepts ~25-30% decode gap vs plain int2/bf16 with CUDA graph."
        ),
    ),
    "speed": GateScenario(
        name="speed",
        min_memory_ratio=0.40,
        min_steady_vs_int2=0.90,
        min_steady_vs_bf16=0.80,
        min_flush_vs_steady=0.55,
        min_request_tok_s_32k=None,
        min_flush_median_tok_s=8.0,
        description="Stricter decode parity for perf-focused deployments.",
    ),
}


def _f(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _row_by_mode(rows: list[dict[str, str]], mode: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("mode") == mode and row.get("server_ok", "").lower() in {"true", "1"}:
            return row
    return None


def resolve_scenario(
    scenario: str | None,
    *,
    min_memory_ratio: float | None,
    min_steady_vs_int2: float | None,
    min_steady_vs_bf16: float | None,
    min_flush_vs_steady: float | None,
    min_request_tok_s_32k: float | None = None,
    min_flush_median_tok_s: float | None = None,
) -> GateScenario:
    if scenario is not None:
        if scenario not in GATE_SCENARIOS:
            known = ", ".join(sorted(GATE_SCENARIOS))
            raise SystemExit(f"Unknown gate scenario {scenario!r}; choose one of: {known}")
        base = GATE_SCENARIOS[scenario]
    else:
        base = GATE_SCENARIOS["balanced"]
    return GateScenario(
        name=base.name,
        min_memory_ratio=(
            min_memory_ratio if min_memory_ratio is not None else base.min_memory_ratio
        ),
        min_steady_vs_int2=(
            min_steady_vs_int2 if min_steady_vs_int2 is not None else base.min_steady_vs_int2
        ),
        min_steady_vs_bf16=(
            min_steady_vs_bf16 if min_steady_vs_bf16 is not None else base.min_steady_vs_bf16
        ),
        min_flush_vs_steady=(
            min_flush_vs_steady
            if min_flush_vs_steady is not None
            else base.min_flush_vs_steady
        ),
        min_request_tok_s_32k=(
            min_request_tok_s_32k
            if min_request_tok_s_32k is not None
            else base.min_request_tok_s_32k
        ),
        min_flush_median_tok_s=(
            min_flush_median_tok_s
            if min_flush_median_tok_s is not None
            else base.min_flush_median_tok_s
        ),
        description=base.description,
    )


def evaluate_gates(
    rows: list[dict[str, str]],
    *,
    min_memory_ratio: float,
    min_steady_vs_int2: float,
    min_steady_vs_bf16: float,
    min_flush_vs_steady: float | None = None,
    min_request_tok_s_32k: float | None = None,
    min_flush_median_tok_s: float | None = None,
) -> list[GateResult]:
    results: list[GateResult] = []
    bf16 = _row_by_mode(rows, "bf16")
    int2 = _row_by_mode(rows, "int2")
    oscar = _row_by_mode(rows, "oscar-int2")

    if oscar is None:
        results.append(GateResult("oscar-int2-present", False, "missing successful oscar-int2 row"))
        return results
    results.append(GateResult("oscar-int2-present", True, "ok"))

    oscar_peak = _f(oscar, "peak_mib_delta")
    bf16_peak = _f(bf16, "peak_mib_delta") if bf16 else None
    oscar_k = _f(oscar, "kv_k_size_gb")
    oscar_v = _f(oscar, "kv_v_size_gb")
    bf16_k = _f(bf16, "kv_k_size_gb") if bf16 else None
    bf16_v = _f(bf16, "kv_v_size_gb") if bf16 else None
    if (
        oscar_k is not None
        and oscar_v is not None
        and bf16_k is not None
        and bf16_v is not None
        and (bf16_k + bf16_v) > 0
    ):
        ratio = (oscar_k + oscar_v) / (bf16_k + bf16_v)
        passed = ratio <= min_memory_ratio
        results.append(
            GateResult(
                "memory-vs-bf16",
                passed,
                f"oscar/bf16 KV pool (K+V) ratio={ratio:.3f} "
                f"(threshold<={min_memory_ratio:.3f})",
            )
        )
    elif oscar_peak is not None and bf16_peak is not None and bf16_peak > 0:
        ratio = oscar_peak / bf16_peak
        passed = ratio <= min_memory_ratio
        results.append(
            GateResult(
                "memory-vs-bf16",
                passed,
                f"oscar/bf16 peak delta ratio={ratio:.3f} (threshold<={min_memory_ratio:.3f})",
            )
        )
    else:
        results.append(
            GateResult(
                "memory-vs-bf16",
                False,
                "missing peak_mib_delta for oscar-int2 and/or bf16",
            )
        )

    oscar_steady = _f(oscar, "decode_steady_median_tok_s")
    int2_steady = _f(int2, "decode_steady_median_tok_s") if int2 else None
    bf16_steady = _f(bf16, "decode_steady_median_tok_s") if bf16 else None

    if min_steady_vs_int2 <= 0:
        results.append(GateResult("steady-vs-int2", True, "skipped by scenario"))
    elif oscar_steady is not None and int2_steady is not None and int2_steady > 0:
        ratio = oscar_steady / int2_steady
        passed = ratio >= min_steady_vs_int2
        results.append(
            GateResult(
                "steady-vs-int2",
                passed,
                f"oscar/int2 steady decode ratio={ratio:.3f} (threshold>={min_steady_vs_int2:.3f})",
            )
        )
    else:
        results.append(
            GateResult(
                "steady-vs-int2",
                True,
                "skipped (missing decode_steady_median_tok_s for oscar-int2 and/or int2)",
            )
        )

    if min_steady_vs_bf16 <= 0:
        results.append(GateResult("steady-vs-bf16", True, "skipped by scenario"))
    elif oscar_steady is not None and bf16_steady is not None and bf16_steady > 0:
        ratio = oscar_steady / bf16_steady
        passed = ratio >= min_steady_vs_bf16
        results.append(
            GateResult(
                "steady-vs-bf16",
                passed,
                f"oscar/bf16 steady decode ratio={ratio:.3f} (threshold>={min_steady_vs_bf16:.3f})",
            )
        )
    else:
        results.append(
            GateResult(
                "steady-vs-bf16",
                False,
                "missing decode_steady_median_tok_s for oscar-int2 and/or bf16",
            )
        )

    kv_selected = _f(oscar, "kv_theory_selected_gib")
    kv_bf16 = _f(oscar, "kv_theory_bf16_gib")
    if kv_selected is not None and kv_bf16 is not None and kv_bf16 > 0:
        ratio = kv_selected / kv_bf16
        passed = ratio <= min_memory_ratio
        results.append(
            GateResult(
                "kv-theory-compression",
                passed,
                f"oscar KV theory ratio={ratio:.3f} (threshold<={min_memory_ratio:.3f})",
            )
        )

    oscar_flush = _f(oscar, "decode_flush_median_tok_s")
    if min_flush_vs_steady is not None:
        if oscar_flush is not None and oscar_steady is not None and oscar_steady > 0:
            ratio = oscar_flush / oscar_steady
            passed = ratio >= min_flush_vs_steady
            results.append(
                GateResult(
                    "flush-vs-steady",
                    passed,
                    f"oscar flush/steady decode ratio={ratio:.3f} "
                    f"(threshold>={min_flush_vs_steady:.3f})",
                )
            )
        else:
            results.append(
                GateResult(
                    "flush-vs-steady",
                    True,
                    "skipped (missing decode_flush_median_tok_s or steady metric)",
                )
            )

    if min_flush_median_tok_s is not None:
        if oscar_flush is not None:
            passed = oscar_flush >= min_flush_median_tok_s
            results.append(
                GateResult(
                    "flush-median-tok-s",
                    passed,
                    f"oscar flush median={oscar_flush:.3f} tok/s "
                    f"(threshold>={min_flush_median_tok_s:.3f})",
                )
            )
        else:
            results.append(
                GateResult(
                    "flush-median-tok-s",
                    True,
                    "skipped (missing decode_flush_median_tok_s)",
                )
            )

    oscar_prefill = _f(oscar, "prefill_tokens")
    oscar_request = _f(oscar, "request_toks_per_sec")
    if min_request_tok_s_32k is not None:
        if (
            oscar_request is not None
            and oscar_prefill is not None
            and oscar_prefill >= 32768
        ):
            passed = oscar_request >= min_request_tok_s_32k
            results.append(
                GateResult(
                    "request-tok-s-32k",
                    passed,
                    f"oscar request={oscar_request:.3f} tok/s "
                    f"(threshold>={min_request_tok_s_32k:.3f})",
                )
            )
        else:
            results.append(
                GateResult(
                    "request-tok-s-32k",
                    True,
                    "skipped (not a 32K prefill row or missing request_toks_per_sec)",
                )
            )

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Check oscar-int2 bench CSV against regression gates.")
    ap.add_argument("csv_path", type=Path, help="bench_*.csv produced by oscar-kv-bench")
    ap.add_argument(
        "--scenario",
        choices=sorted(GATE_SCENARIOS.keys()),
        default="balanced",
        help="Preset threshold bundle (default: balanced).",
    )
    ap.add_argument(
        "--min-memory-ratio",
        type=float,
        default=None,
        help="Override scenario: require oscar peak/theoretical KV <= this fraction of bf16.",
    )
    ap.add_argument(
        "--min-steady-vs-int2",
        type=float,
        default=None,
        help="Override scenario: require oscar steady decode >= this fraction of plain int2.",
    )
    ap.add_argument(
        "--min-steady-vs-bf16",
        type=float,
        default=None,
        help="Override scenario: require oscar steady decode >= this fraction of bf16.",
    )
    ap.add_argument(
        "--min-flush-vs-steady",
        type=float,
        default=None,
        help="Override scenario: require oscar flush/steady decode ratio >= this value.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    scenario = resolve_scenario(
        args.scenario,
        min_memory_ratio=args.min_memory_ratio,
        min_steady_vs_int2=args.min_steady_vs_int2,
        min_steady_vs_bf16=args.min_steady_vs_bf16,
        min_flush_vs_steady=args.min_flush_vs_steady,
    )

    with args.csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    results = evaluate_gates(
        rows,
        min_memory_ratio=scenario.min_memory_ratio,
        min_steady_vs_int2=scenario.min_steady_vs_int2,
        min_steady_vs_bf16=scenario.min_steady_vs_bf16,
        min_flush_vs_steady=scenario.min_flush_vs_steady,
        min_request_tok_s_32k=scenario.min_request_tok_s_32k,
        min_flush_median_tok_s=scenario.min_flush_median_tok_s,
    )

    print(f"# regression gate: {args.csv_path}", flush=True)
    print(f"# scenario: {scenario.name} — {scenario.description}", flush=True)
    all_ok = True
    for item in results:
        status = "PASS" if item.passed else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}", flush=True)
        all_ok = all_ok and item.passed

    payload = {
        "csv_path": str(args.csv_path),
        "scenario": scenario.name,
        "passed": all_ok,
        "checks": [
            {"name": r.name, "passed": r.passed, "detail": r.detail} for r in results
        ],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
