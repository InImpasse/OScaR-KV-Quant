"""Compare longrun JSON reports against stability, memory, and speed gates."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oscar_kv_quant.log_metrics import parse_server_log


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def _f(data: dict[str, Any], key: str) -> float | None:
    raw = data.get(key)
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _request_tok_s(data: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for row in data.get("requests") or []:
        if not row.get("ok", False):
            continue
        tok_s = row.get("tok_s")
        if tok_s is None:
            continue
        values.append(float(tok_s))
    return values


def _tail_median(values: list[float]) -> float | None:
    if not values:
        return None
    tail = values[len(values) // 2 :]
    return statistics.median(tail)


def _last_quarter_median(values: list[float]) -> float | None:
    if not values:
        return None
    start = max(0, len(values) - max(1, len(values) // 4))
    return statistics.median(values[start:])


def _ratio_check(
    name: str,
    numerator: float | None,
    denominator: float | None,
    threshold: float,
    *,
    compare: str,
    label: str,
    strict_less: bool = False,
    ratio_subject: str = "oscar/baseline",
) -> GateResult:
    if numerator is None or denominator is None or denominator <= 0:
        return GateResult(name, False, f"missing {label} for oscar and/or baseline")
    ratio = numerator / denominator
    if compare == ">=":
        passed = ratio >= threshold
        op = ">="
    elif compare == "<=":
        passed = ratio < threshold if strict_less else ratio <= threshold
        op = "<" if strict_less else "<="
    else:
        raise ValueError(compare)
    return GateResult(
        name,
        passed,
        f"{ratio_subject} {label} ratio={ratio:.3f} (threshold{op}{threshold:.3f})",
    )


def evaluate_longrun_gate(
    oscar: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_kv_ratio: float = 0.35,
    min_request_ratio: float = 1.0,
    min_tail_request_ratio: float = 1.0,
    min_decode_ratio: float = 1.0,
    require_peak_lower: bool = True,
    require_no_log_errors: bool = True,
    min_second_half_vs_first_half: float = 0.90,
) -> list[GateResult]:
    results: list[GateResult] = []

    for key in ("profile", "prefill_tokens", "max_new_tokens", "num_requests"):
        if oscar.get(key) != baseline.get(key):
            results.append(
                GateResult(
                    f"same-{key.replace('_', '-')}",
                    False,
                    f"oscar={oscar.get(key)!r} baseline={baseline.get(key)!r}",
                )
            )
        else:
            results.append(
                GateResult(
                    f"same-{key.replace('_', '-')}",
                    True,
                    f"value={oscar.get(key)!r}",
                )
            )

    for label, data in (("oscar", oscar), ("baseline", baseline)):
        ok = bool(data.get("ok"))
        failures = int(data.get("failures") or 0)
        completed = int(data.get("completed_requests") or 0)
        expected = int(data.get("num_requests") or 0)
        passed = ok and failures == 0 and completed == expected and expected > 0
        results.append(
            GateResult(
                f"{label}-stable",
                passed,
                f"ok={ok} failures={failures} completed={completed}/{expected}",
            )
        )
        if require_no_log_errors:
            log_errors = data.get("log_errors") or []
            results.append(
                GateResult(
                    f"{label}-log-clean",
                    len(log_errors) == 0,
                    f"log_errors={len(log_errors)}",
                )
            )

    oscar_peak = _f(oscar, "peak_mib_total")
    baseline_peak = _f(baseline, "peak_mib_total")
    if require_peak_lower:
        if oscar_peak is None or baseline_peak is None:
            results.append(GateResult("peak-memory-lower", False, "missing peak_mib_total"))
        else:
            results.append(
                GateResult(
                    "peak-memory-lower",
                    oscar_peak < baseline_peak,
                    f"oscar peak={oscar_peak:.0f} MiB baseline peak={baseline_peak:.0f} MiB",
                )
            )

    oscar_kv = _f(oscar, "selected_vs_bf16_kv_ratio")
    if oscar_kv is not None:
        results.append(
            GateResult(
                "kv-compression",
                oscar_kv <= max_kv_ratio,
                f"oscar selected_vs_bf16_kv_ratio={oscar_kv:.3f} "
                f"(threshold<={max_kv_ratio:.3f})",
            )
        )
    else:
        oscar_k = _f(oscar, "kv_k_size_gb")
        oscar_v = _f(oscar, "kv_v_size_gb")
        baseline_k = _f(baseline, "kv_k_size_gb")
        baseline_v = _f(baseline, "kv_v_size_gb")
        results.append(
            _ratio_check(
                "kv-compression",
                None if oscar_k is None or oscar_v is None else oscar_k + oscar_v,
                None
                if baseline_k is None or baseline_v is None
                else baseline_k + baseline_v,
                max_kv_ratio,
                compare="<=",
                label="KV K+V GB",
            )
        )

    results.append(
        _ratio_check(
            "request-median-speed",
            _f(oscar, "request_tok_s_median"),
            _f(baseline, "request_tok_s_median"),
            min_request_ratio,
            compare=">=",
            label="request median tok/s",
        )
    )

    oscar_tail = _tail_median(_request_tok_s(oscar))
    baseline_tail = _tail_median(_request_tok_s(baseline))
    results.append(
        _ratio_check(
            "request-tail-speed",
            oscar_tail,
            baseline_tail,
            min_tail_request_ratio,
            compare=">=",
            label="tail-half request tok/s median",
        )
    )

    results.append(
        _ratio_check(
            "decode-steady-speed",
            _f(oscar, "server_decode_steady_median_tok_s"),
            _f(baseline, "server_decode_steady_median_tok_s"),
            min_decode_ratio,
            compare=">=",
            label="server steady decode tok/s",
        )
    )

    oscar_first_half = _f(oscar, "request_tok_s_first_half_median")
    oscar_second_half = _f(oscar, "request_tok_s_second_half_median")
    if oscar_first_half is not None and oscar_second_half is not None:
        results.append(
            _ratio_check(
                "oscar-second-half-stability",
                oscar_second_half,
                oscar_first_half,
                min_second_half_vs_first_half,
                compare=">=",
                label="request median tok/s",
                ratio_subject="oscar second/first half",
            )
        )

    oscar_values = _request_tok_s(oscar)
    oscar_last_quarter = _last_quarter_median(oscar_values)
    if oscar_first_half is not None and oscar_last_quarter is not None:
        results.append(
            _ratio_check(
                "oscar-last-quarter-stability",
                oscar_last_quarter,
                oscar_first_half,
                min_second_half_vs_first_half,
                compare=">=",
                label="request median tok/s",
                ratio_subject="oscar last-quarter/first-half",
            )
        )

    oscar_cache_ratio = _f(oscar, "cached_prefill_cache_ratio_median")
    baseline_cache_ratio = _f(baseline, "cached_prefill_cache_ratio_median")
    if oscar_cache_ratio is not None and baseline_cache_ratio is not None:
        results.append(
            GateResult(
                "cache-reuse-present",
                oscar_cache_ratio > 0.0 and baseline_cache_ratio > 0.0,
                f"oscar cached-prefill ratio={oscar_cache_ratio:.4f} "
                f"baseline cached-prefill ratio={baseline_cache_ratio:.4f}",
            )
        )

    return results


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def evaluate_longrun_many_gate(
    oscars: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    *,
    max_kv_ratio: float = 0.35,
    min_request_ratio: float = 1.0,
    min_tail_request_ratio: float = 1.0,
    min_decode_ratio: float = 1.0,
    require_peak_lower: bool = True,
    require_no_log_errors: bool = True,
    min_second_half_vs_first_half: float = 0.90,
) -> list[GateResult]:
    results: list[GateResult] = []
    if not oscars:
        return [GateResult("oscar-runs-present", False, "no oscar reports")]
    if not baselines:
        return [GateResult("baseline-runs-present", False, "no baseline reports")]

    ref = oscars[0]
    workload_keys = ("profile", "prefill_tokens", "max_new_tokens", "num_requests")
    for key in workload_keys:
        values = [run.get(key) for run in oscars + baselines]
        same = all(v == values[0] for v in values)
        results.append(
            GateResult(
                f"same-{key.replace('_', '-')}",
                same,
                f"values={values}",
            )
        )

    for label, runs in (("oscar", oscars), ("baseline", baselines)):
        bad = [
            i
            for i, data in enumerate(runs, start=1)
            if not bool(data.get("ok"))
            or int(data.get("failures") or 0) != 0
            or int(data.get("completed_requests") or 0) != int(data.get("num_requests") or 0)
            or int(data.get("num_requests") or 0) <= 0
        ]
        results.append(
            GateResult(
                f"{label}-stable-all",
                not bad,
                f"runs={len(runs)} bad_indices={bad}",
            )
        )
        if require_no_log_errors:
            with_errors = [
                i
                for i, data in enumerate(runs, start=1)
                if len(data.get("log_errors") or []) > 0
            ]
            results.append(
                GateResult(
                    f"{label}-log-clean-all",
                    not with_errors,
                    f"runs={len(runs)} with_errors={with_errors}",
                )
            )

    oscar_peak_max = _max_present([_f(r, "peak_mib_total") for r in oscars])
    baseline_peak_min = _min_present([_f(r, "peak_mib_total") for r in baselines])
    if require_peak_lower:
        passed = (
            oscar_peak_max is not None
            and baseline_peak_min is not None
            and oscar_peak_max < baseline_peak_min
        )
        results.append(
            GateResult(
                "peak-memory-lower-all",
                passed,
                f"oscar max peak={oscar_peak_max} MiB baseline min peak={baseline_peak_min} MiB",
            )
        )

    oscar_kv_max = _max_present([_f(r, "selected_vs_bf16_kv_ratio") for r in oscars])
    if oscar_kv_max is None:
        oscar_kv_vals = [
            None if _f(r, "kv_k_size_gb") is None or _f(r, "kv_v_size_gb") is None
            else (_f(r, "kv_k_size_gb") or 0.0) + (_f(r, "kv_v_size_gb") or 0.0)
            for r in oscars
        ]
        baseline_kv_vals = [
            None if _f(r, "kv_k_size_gb") is None or _f(r, "kv_v_size_gb") is None
            else (_f(r, "kv_k_size_gb") or 0.0) + (_f(r, "kv_v_size_gb") or 0.0)
            for r in baselines
        ]
        oscar_kv_max_abs = _max_present(oscar_kv_vals)
        baseline_kv_min_abs = _min_present(baseline_kv_vals)
        oscar_kv_max = (
            None
            if oscar_kv_max_abs is None or baseline_kv_min_abs in (None, 0.0)
            else oscar_kv_max_abs / baseline_kv_min_abs
        )
    results.append(
        GateResult(
            "kv-compression-all",
            oscar_kv_max is not None and oscar_kv_max <= max_kv_ratio,
            f"oscar max KV ratio={oscar_kv_max} (threshold<={max_kv_ratio:.3f})",
        )
    )

    def ratio_result(
        name: str,
        oscar_key: str,
        baseline_key: str,
        threshold: float,
        label: str,
    ) -> GateResult:
        oscar_min = _min_present([_f(r, oscar_key) for r in oscars])
        baseline_max = _max_present([_f(r, baseline_key) for r in baselines])
        ratio = (
            None
            if oscar_min is None or baseline_max in (None, 0.0)
            else oscar_min / baseline_max
        )
        return GateResult(
            name,
            ratio is not None and ratio >= threshold,
            f"worst oscar/baseline {label} ratio={ratio} (threshold>={threshold:.3f})",
        )

    results.append(
        ratio_result(
            "request-median-speed-all",
            "request_tok_s_median",
            "request_tok_s_median",
            min_request_ratio,
            "request median tok/s",
        )
    )

    oscar_tail_mins = [_tail_median(_request_tok_s(r)) for r in oscars]
    baseline_tail_max = _max_present([_tail_median(_request_tok_s(r)) for r in baselines])
    oscar_tail_min = _min_present(oscar_tail_mins)
    tail_ratio = (
        None
        if oscar_tail_min is None or baseline_tail_max in (None, 0.0)
        else oscar_tail_min / baseline_tail_max
    )
    results.append(
        GateResult(
            "request-tail-speed-all",
            tail_ratio is not None and tail_ratio >= min_tail_request_ratio,
            f"worst oscar/baseline tail-half request tok/s ratio={tail_ratio} "
            f"(threshold>={min_tail_request_ratio:.3f})",
        )
    )

    results.append(
        ratio_result(
            "decode-steady-speed-all",
            "server_decode_steady_median_tok_s",
            "server_decode_steady_median_tok_s",
            min_decode_ratio,
            "server steady decode tok/s",
        )
    )

    stability_ratios: list[float | None] = []
    last_quarter_ratios: list[float | None] = []
    for run in oscars:
        first = _f(run, "request_tok_s_first_half_median")
        second = _f(run, "request_tok_s_second_half_median")
        stability_ratios.append(
            None if first in (None, 0.0) or second is None else second / first
        )
        last_quarter = _last_quarter_median(_request_tok_s(run))
        last_quarter_ratios.append(
            None if first in (None, 0.0) or last_quarter is None else last_quarter / first
        )
    min_stability = _min_present(stability_ratios)
    if min_stability is not None:
        results.append(
            GateResult(
                "oscar-second-half-stability-all",
                min_stability >= min_second_half_vs_first_half,
                f"worst oscar second/first half request median ratio={min_stability} "
                f"(threshold>={min_second_half_vs_first_half:.3f})",
            )
        )
    min_last_quarter_stability = _min_present(last_quarter_ratios)
    if min_last_quarter_stability is not None:
        results.append(
            GateResult(
                "oscar-last-quarter-stability-all",
                min_last_quarter_stability >= min_second_half_vs_first_half,
                f"worst oscar last-quarter/first-half request median ratio="
                f"{min_last_quarter_stability} "
                f"(threshold>={min_second_half_vs_first_half:.3f})",
            )
        )

    oscar_cache_min = _min_present([_f(r, "cached_prefill_cache_ratio_median") for r in oscars])
    if oscar_cache_min is not None:
        results.append(
            GateResult(
                "cache-reuse-present-all",
                oscar_cache_min > 0.0,
                f"oscar min cached-prefill ratio={oscar_cache_min:.4f}",
            )
        )

    return results


def _max_present(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _min_present(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return data


def _refresh_log_metrics(data: dict[str, Any]) -> dict[str, Any]:
    log_path_raw = data.get("server_log_path")
    if not log_path_raw:
        return data
    log_path = Path(log_path_raw)
    metrics = parse_server_log(
        log_path,
        measurement_requests=int(data.get("completed_requests") or data.get("num_requests") or 1),
        decode_tokens_per_request=(
            int(data["max_new_tokens"]) if data.get("max_new_tokens") is not None else None
        ),
    )
    out = dict(data)
    out["server_decode_steady_median_tok_s"] = metrics.decode_steady_median_tok_s
    out["server_decode_max_tok_s"] = metrics.decode_max_tok_s
    out["prefill_median_tok_s"] = metrics.prefill_median_tok_s
    out["cached_prefill_new_median_tokens"] = metrics.cached_prefill_new_median_tokens
    out["cached_prefill_cached_median_tokens"] = metrics.cached_prefill_cached_median_tokens
    out["cached_prefill_cache_ratio_median"] = metrics.cached_prefill_cache_ratio_median
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare oscar-int2 and baseline longrun reports.")
    ap.add_argument("--oscar", type=Path, default=None, help="oscar-int2 longrun.json")
    ap.add_argument("--baseline", type=Path, default=None, help="bf16/fp16 longrun.json")
    ap.add_argument("--oscar-many", type=Path, nargs="+", default=None, help="multiple oscar longrun.json files")
    ap.add_argument("--baseline-many", type=Path, nargs="+", default=None, help="multiple baseline longrun.json files")
    ap.add_argument("--max-kv-ratio", type=float, default=0.35)
    ap.add_argument("--min-request-ratio", type=float, default=1.0)
    ap.add_argument("--min-tail-request-ratio", type=float, default=1.0)
    ap.add_argument("--min-decode-ratio", type=float, default=1.0)
    ap.add_argument("--min-second-half-vs-first-half", type=float, default=0.90)
    ap.add_argument("--allow-peak-equal-or-higher", action="store_true")
    ap.add_argument("--allow-log-errors", action="store_true")
    ap.add_argument(
        "--refresh-log-metrics",
        action="store_true",
        help="Recompute log-derived decode/cache metrics from server_log_path before gating.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.oscar_many or args.baseline_many:
        if not args.oscar_many or not args.baseline_many:
            raise SystemExit("--oscar-many and --baseline-many must be used together")
        results = evaluate_longrun_many_gate(
            [
                _refresh_log_metrics(_load_json(p)) if args.refresh_log_metrics else _load_json(p)
                for p in args.oscar_many
            ],
            [
                _refresh_log_metrics(_load_json(p)) if args.refresh_log_metrics else _load_json(p)
                for p in args.baseline_many
            ],
            max_kv_ratio=args.max_kv_ratio,
            min_request_ratio=args.min_request_ratio,
            min_tail_request_ratio=args.min_tail_request_ratio,
            min_decode_ratio=args.min_decode_ratio,
            require_peak_lower=not args.allow_peak_equal_or_higher,
            require_no_log_errors=not args.allow_log_errors,
            min_second_half_vs_first_half=args.min_second_half_vs_first_half,
        )
        oscar_label = ",".join(str(p) for p in args.oscar_many)
        baseline_label = ",".join(str(p) for p in args.baseline_many)
    else:
        if args.oscar is None or args.baseline is None:
            raise SystemExit("provide --oscar/--baseline or --oscar-many/--baseline-many")
        results = evaluate_longrun_gate(
            _refresh_log_metrics(_load_json(args.oscar))
            if args.refresh_log_metrics
            else _load_json(args.oscar),
            _refresh_log_metrics(_load_json(args.baseline))
            if args.refresh_log_metrics
            else _load_json(args.baseline),
            max_kv_ratio=args.max_kv_ratio,
            min_request_ratio=args.min_request_ratio,
            min_tail_request_ratio=args.min_tail_request_ratio,
            min_decode_ratio=args.min_decode_ratio,
            require_peak_lower=not args.allow_peak_equal_or_higher,
            require_no_log_errors=not args.allow_log_errors,
            min_second_half_vs_first_half=args.min_second_half_vs_first_half,
        )
        oscar_label = str(args.oscar)
        baseline_label = str(args.baseline)

    print(f"# longrun gate: oscar={oscar_label}", flush=True)
    print(f"# baseline={baseline_label}", flush=True)
    all_ok = True
    for item in results:
        status = "PASS" if item.passed else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}", flush=True)
        all_ok = all_ok and item.passed

    payload = {
        "oscar": oscar_label,
        "baseline": baseline_label,
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
