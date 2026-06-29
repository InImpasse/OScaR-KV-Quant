#!/usr/bin/env python3
"""Measure first-token and per-token decode latency through llama-server SSE."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from urllib import request


PRESET_TOKENS = {
    "short": 512,
    "medium": 2048,
    "long": 8192,
    "16k": 16384,
    "32k": 32768,
}


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_prompt(server: str, prompt_tokens: int, timeout: float) -> str:
    seed = (
        "The following is deterministic benchmark filler. "
        "Repeat the pattern and continue without solving a task. "
    )
    prompt = seed
    while True:
        res = _post_json(
            f"{server}/tokenize",
            {"content": prompt, "add_special": True, "parse_special": True},
            timeout,
        )
        tokens = res.get("tokens", [])
        if len(tokens) >= prompt_tokens:
            return _post_json(
                f"{server}/detokenize",
                {"tokens": tokens[:prompt_tokens], "special": False},
                timeout,
            ).get("content", prompt)
        missing = prompt_tokens - len(tokens)
        prompt += (" benchmark-context" * max(64, missing // 2))


def parse_sse_line(line: bytes) -> dict | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    data = text[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def token_text(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def measure_stream(server: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": "llama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
        "cache_prompt": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{server}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start = time.perf_counter()
    token_times: list[float] = []
    timings: dict = {}
    with request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            now = time.perf_counter()
            chunk = parse_sse_line(raw)
            if chunk is None:
                continue
            if "timings" in chunk:
                timings = chunk["timings"]
            if token_text(chunk):
                token_times.append(now)

    first = token_times[0] - start if token_times else None
    intervals = [cur - prev for prev, cur in zip(token_times, token_times[1:]) if cur > prev]
    steady = None
    p95 = None
    if intervals:
        rates = [1.0 / value for value in intervals if value > 0]
        steady = statistics.median(rates) if rates else None
        if len(rates) == 1:
            p95 = rates[0]
        elif rates:
            p95 = sorted(rates)[max(0, int(len(rates) * 0.05) - 1)]

    return {
        "decode_first_tok_s": "" if first is None or first <= 0 else 1.0 / first,
        "decode_steady_median_tok_s": steady or "",
        "decode_steady_p95_tok_s": p95 or "",
        "stream_tokens": len(token_times),
        "server_predicted_per_second": timings.get("predicted_per_second", ""),
        "server_prompt_per_second": timings.get("prompt_per_second", ""),
    }


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8033")
    parser.add_argument("--preset", required=True, choices=PRESET_TOKENS)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt_tokens = PRESET_TOKENS[args.preset]
    prompt = make_prompt(args.server, prompt_tokens, args.timeout)
    result = measure_stream(args.server, prompt, args.max_tokens, args.timeout)
    result.update({
        "preset": args.preset,
        "mode": args.mode,
        "prefill_tokens": prompt_tokens,
        "max_new_tokens": args.max_tokens,
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow({key: fmt(value) for key, value in result.items()})

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
