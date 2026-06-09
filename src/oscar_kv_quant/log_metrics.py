"""Parse SGLang server logs into structured benchmark metrics."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RequestLogWindow:
    prefill_new_tokens: int | None
    prefill_cached_tokens: int | None
    prefill_tok_s: float | None
    prefill_interval_est_s: float | None
    prefill_wall_est_s: float | None
    decode_first_tok_s: float | None
    decode_steady_median_tok_s: float | None
    decode_flush_median_tok_s: float | None
    decode_samples: int


@dataclass
class ServerLogMetrics:
    decode_throughputs: list[float]
    prefill_throughputs: list[float]
    decode_max_tok_s: float | None
    decode_first_tok_s: float | None
    decode_steady_median_tok_s: float | None
    decode_steady_p95_tok_s: float | None
    decode_flush_median_tok_s: float | None
    prefill_median_tok_s: float | None
    prefill_wall_s: float | None
    decode_wall_s: float | None
    flush_step_fraction: float | None
    effective_decode_tok_s: float | None
    prefill_new_tokens: list[int]
    prefill_cached_tokens: list[int]
    cached_prefill_new_median_tokens: float | None
    cached_prefill_cached_median_tokens: float | None
    cached_prefill_cache_ratio_median: float | None
    kv_pool_tokens: int | None
    kv_k_size_gb: float | None
    kv_v_size_gb: float | None
    max_total_num_tokens: int | None
    hp_prefix_pool_tokens: int | None
    unified_mixed_kv: bool
    request_windows: list[RequestLogWindow]


_DECODE_RE = re.compile(
    r"Decode batch,.*gen throughput \(token/s\): ([0-9.]+)"
)
_PREFILL_RE = re.compile(
    r"Prefill batch,.*input throughput \(token/s\): ([0-9.]+)"
)
_PREFILL_TOKENS_RE = re.compile(
    r"Prefill batch,.*#new-token: (\d+), #cached-token: (\d+)"
)
_KV_ALLOC_RE = re.compile(
    r"KV Cache is allocated\. #tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB"
)
_MAX_TOTAL_RE = re.compile(r"max_total_num_tokens=(\d+)")
_HP_PREFIX_POOL_RE = re.compile(r"hp_prefix_pool_tokens=(\d+)")
_UNIFIED_MIXED_RE = re.compile(r"Enable unified mixed KV \(int2\)")
_LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def _parse_log_timestamp(line: str) -> datetime | None:
    m = _LOG_TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _classify_decode_throughputs(
    throughputs: list[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return first, steady median, steady p95, flush-step median decode tok/s."""
    if not throughputs:
        return None, None, None, None
    first = throughputs[0]
    if len(throughputs) == 1:
        return first, first, first, None

    # Drop the first decode step (often cold) and classify low outliers as flush steps.
    tail = throughputs[1:]
    if not tail:
        return first, None, None, None

    # Decode logs can be bimodal for mixed-KV: low periodic flush/scheduler
    # samples interleave with normal decode samples. A plain median can land in
    # the low cluster when short generations emit only a few log lines, so use
    # the upper quartile as a robust estimate of the high-throughput plateau.
    baseline = _percentile(tail, 75) or statistics.median(tail)
    flush_threshold = max(1.0, baseline * 0.55)
    steady = [v for v in tail if v >= flush_threshold]
    flush = [v for v in tail if v < flush_threshold]

    steady_median = statistics.median(steady) if steady else None
    steady_p95 = _percentile(steady, 95)
    flush_median = statistics.median(flush) if flush else None
    return first, steady_median, steady_p95, flush_median


def _flush_step_fraction(throughputs: list[float]) -> float | None:
    if len(throughputs) <= 1:
        return None
    tail = throughputs[1:]
    if not tail:
        return None
    baseline = _percentile(tail, 75) or statistics.median(tail)
    flush_threshold = max(1.0, baseline * 0.55)
    flush_count = sum(1 for v in tail if v < flush_threshold)
    return flush_count / len(tail)


@dataclass(frozen=True)
class _RequestWindow:
    prefill_times: tuple[datetime, ...]
    decode_times: tuple[datetime, ...]


@dataclass(frozen=True)
class _RequestMetricWindow:
    prefill_new_tokens: tuple[int, ...]
    prefill_cached_tokens: tuple[int, ...]
    prefill_throughputs: tuple[float, ...]
    decode_throughputs: tuple[float, ...]


def _split_log_into_requests(
    prefill_times: list[datetime],
    decode_times: list[datetime],
) -> list[_RequestWindow]:
    """Group timestamped prefill/decode bursts into per-request windows."""
    if not prefill_times and not decode_times:
        return []

    events = sorted(
        [(ts, "prefill") for ts in prefill_times]
        + [(ts, "decode") for ts in decode_times],
        key=lambda item: item[0],
    )
    requests: list[_RequestWindow] = []
    cur_prefill: list[datetime] = []
    cur_decode: list[datetime] = []
    for _ts, kind in events:
        if kind == "prefill":
            if cur_decode:
                requests.append(
                    _RequestWindow(tuple(cur_prefill), tuple(cur_decode))
                )
                cur_prefill = []
                cur_decode = []
            cur_prefill.append(_ts)
        else:
            cur_decode.append(_ts)
    if cur_prefill or cur_decode:
        requests.append(_RequestWindow(tuple(cur_prefill), tuple(cur_decode)))
    return requests


def _split_metrics_into_requests(
    events: list[tuple[datetime | None, str, tuple[int | float, ...]]],
) -> list[_RequestMetricWindow]:
    # SGLang logs ``input throughput`` on prefill line N as:
    #     tokens from prefill N-1 / wall time between prefill N-1 and N.
    # Therefore the throughput value seen on a new prefill event belongs to the
    # previous request window, while that event's token counts start the next
    # window.
    requests: list[_RequestMetricWindow] = []
    cur_new: list[int] = []
    cur_cached: list[int] = []
    cur_prefill_tps: list[float] = []
    cur_decode_tps: list[float] = []
    for _ts, kind, values in events:
        if kind == "prefill":
            new_tokens, cached_tokens, throughput = values
            if cur_new or cur_cached or cur_prefill_tps or cur_decode_tps:
                cur_prefill_tps.append(float(throughput))
                requests.append(
                    _RequestMetricWindow(
                        tuple(cur_new),
                        tuple(cur_cached),
                        tuple(cur_prefill_tps),
                        tuple(cur_decode_tps),
                    )
                )
                cur_new = []
                cur_cached = []
                cur_prefill_tps = []
                cur_decode_tps = []
            cur_new.append(int(new_tokens))
            cur_cached.append(int(cached_tokens))
        else:
            (throughput,) = values
            cur_decode_tps.append(float(throughput))
    if cur_new or cur_cached or cur_prefill_tps or cur_decode_tps:
        requests.append(
            _RequestMetricWindow(
                tuple(cur_new),
                tuple(cur_cached),
                tuple(cur_prefill_tps),
                tuple(cur_decode_tps),
            )
        )
    return requests


def _prefill_wall_estimate(window: _RequestMetricWindow) -> float | None:
    if not window.prefill_throughputs:
        return None
    total = 0.0
    found = False
    for new_tokens, throughput in zip(
        window.prefill_new_tokens, window.prefill_throughputs
    ):
        if throughput <= 0:
            continue
        total += float(new_tokens) / float(throughput)
        found = True
    return total if found else None


def _request_log_windows(
    events: list[tuple[datetime | None, str, tuple[int | float, ...]]],
) -> list[RequestLogWindow]:
    out: list[RequestLogWindow] = []
    for window in _split_metrics_into_requests(events):
        first, steady, _p95, flush = _classify_decode_throughputs(
            list(window.decode_throughputs)
        )
        out.append(
            RequestLogWindow(
                prefill_new_tokens=(
                    int(sum(window.prefill_new_tokens))
                    if window.prefill_new_tokens
                    else None
                ),
                prefill_cached_tokens=(
                    int(sum(window.prefill_cached_tokens))
                    if window.prefill_cached_tokens
                    else None
                ),
                prefill_tok_s=(
                    statistics.median(window.prefill_throughputs)
                    if window.prefill_throughputs
                    else None
                ),
                prefill_interval_est_s=_prefill_wall_estimate(window),
                # Back-compat alias: SGLang's prefill throughput is measured
                # over the interval between prefill stat reports, not isolated
                # prefill-forward wall time.
                prefill_wall_est_s=_prefill_wall_estimate(window),
                decode_first_tok_s=first,
                decode_steady_median_tok_s=steady,
                decode_flush_median_tok_s=flush,
                decode_samples=len(window.decode_throughputs),
            )
        )
    return out


def _wall_span_seconds(times: tuple[datetime, ...]) -> float | None:
    if len(times) < 2:
        return None
    span = max(0.0, (times[-1] - times[0]).total_seconds())
    # Server logs are whole-second; multiple decode batches in the same second
    # still represent measurable work.
    if span <= 0.0:
        return 1.0
    return span


def _wall_times_for_measurement(
    prefill_times: list[datetime],
    decode_times: list[datetime],
    *,
    measurement_requests: int,
) -> tuple[float | None, float | None]:
    requests = _split_log_into_requests(prefill_times, decode_times)
    if not requests:
        return None, None
    take = max(1, measurement_requests)
    window = requests[-take:]
    prefill_walls = [_wall_span_seconds(r.prefill_times) for r in window]
    decode_walls = [_wall_span_seconds(r.decode_times) for r in window]
    prefill_walls = [w for w in prefill_walls if w is not None]
    decode_walls = [w for w in decode_walls if w is not None]
    prefill_wall = statistics.mean(prefill_walls) if prefill_walls else None
    decode_wall = statistics.mean(decode_walls) if decode_walls else None
    return prefill_wall, decode_wall


def parse_server_log(
    path: Path,
    *,
    measurement_requests: int = 1,
    decode_tokens_per_request: int | None = None,
) -> ServerLogMetrics:
    text = path.read_text(errors="replace") if path.is_file() else ""
    decode_vals: list[float] = []
    prefill_vals: list[float] = []
    prefill_new_tokens: list[int] = []
    prefill_cached_tokens: list[int] = []
    prefill_times: list[datetime] = []
    decode_times: list[datetime] = []
    request_events: list[tuple[datetime | None, str, tuple[int | float, ...]]] = []
    kv_pool_tokens: int | None = None
    kv_k_size_gb: float | None = None
    kv_v_size_gb: float | None = None
    max_total_num_tokens: int | None = None
    hp_prefix_pool_tokens: int | None = None
    unified_mixed_kv = False

    for line in text.splitlines():
        ts = _parse_log_timestamp(line)
        prefill_token_pair: tuple[int, int] | None = None
        if m := _PREFILL_TOKENS_RE.search(line):
            prefill_token_pair = (int(m.group(1)), int(m.group(2)))
            prefill_new_tokens.append(prefill_token_pair[0])
            prefill_cached_tokens.append(prefill_token_pair[1])
        if m := _DECODE_RE.search(line):
            try:
                throughput = float(m.group(1))
                decode_vals.append(throughput)
                request_events.append((ts, "decode", (throughput,)))
                if ts is not None:
                    decode_times.append(ts)
            except ValueError:
                pass
            continue
        if m := _PREFILL_RE.search(line):
            try:
                throughput = float(m.group(1))
                prefill_vals.append(throughput)
                if prefill_token_pair is None:
                    prefill_token_pair = (0, 0)
                request_events.append(
                    (
                        ts,
                        "prefill",
                        (prefill_token_pair[0], prefill_token_pair[1], throughput),
                    )
                )
                if ts is not None:
                    prefill_times.append(ts)
            except ValueError:
                pass
            continue
        if m := _KV_ALLOC_RE.search(line):
            kv_pool_tokens = int(m.group(1))
            kv_k_size_gb = float(m.group(2))
            kv_v_size_gb = float(m.group(3))
        if m := _MAX_TOTAL_RE.search(line):
            max_total_num_tokens = int(m.group(1))
        if m := _HP_PREFIX_POOL_RE.search(line):
            hp_prefix_pool_tokens = int(m.group(1))
        if _UNIFIED_MIXED_RE.search(line):
            unified_mixed_kv = True

    first, steady_med, steady_p95, flush_med = _classify_decode_throughputs(decode_vals)
    prefill_med = statistics.median(prefill_vals[2:]) if len(prefill_vals) > 2 else (
        statistics.median(prefill_vals) if prefill_vals else None
    )
    flush_frac = _flush_step_fraction(decode_vals)
    prefill_wall, decode_wall = _wall_times_for_measurement(
        prefill_times,
        decode_times,
        measurement_requests=measurement_requests,
    )
    effective_decode: float | None = None
    if (
        decode_wall is not None
        and decode_wall > 0
        and decode_tokens_per_request is not None
        and decode_tokens_per_request > 0
    ):
        effective_decode = decode_tokens_per_request / decode_wall

    cached_pairs = [
        (new, cached)
        for new, cached in zip(prefill_new_tokens, prefill_cached_tokens)
        if cached > 0
    ]
    cached_new_median = (
        statistics.median([new for new, _cached in cached_pairs])
        if cached_pairs
        else None
    )
    cached_cached_median = (
        statistics.median([cached for _new, cached in cached_pairs])
        if cached_pairs
        else None
    )
    cached_ratio_median = (
        statistics.median(
            [
                cached / (new + cached)
                for new, cached in cached_pairs
                if new + cached > 0
            ]
        )
        if cached_pairs
        else None
    )

    return ServerLogMetrics(
        decode_throughputs=decode_vals,
        prefill_throughputs=prefill_vals,
        decode_max_tok_s=max(decode_vals) if decode_vals else None,
        decode_first_tok_s=first,
        decode_steady_median_tok_s=steady_med,
        decode_steady_p95_tok_s=steady_p95,
        decode_flush_median_tok_s=flush_med,
        prefill_median_tok_s=prefill_med,
        prefill_wall_s=prefill_wall,
        decode_wall_s=decode_wall,
        flush_step_fraction=flush_frac,
        effective_decode_tok_s=effective_decode,
        prefill_new_tokens=prefill_new_tokens,
        prefill_cached_tokens=prefill_cached_tokens,
        cached_prefill_new_median_tokens=cached_new_median,
        cached_prefill_cached_median_tokens=cached_cached_median,
        cached_prefill_cache_ratio_median=cached_ratio_median,
        kv_pool_tokens=kv_pool_tokens,
        kv_k_size_gb=kv_k_size_gb,
        kv_v_size_gb=kv_v_size_gb,
        max_total_num_tokens=max_total_num_tokens,
        hp_prefix_pool_tokens=hp_prefix_pool_tokens,
        unified_mixed_kv=unified_mixed_kv,
        request_windows=_request_log_windows(request_events),
    )
