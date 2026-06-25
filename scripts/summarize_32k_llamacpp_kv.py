#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from estimate_kv_cache import estimate_bytes, mib, model_dims, read_metadata  # noqa: E402


LABEL_RE = re.compile(r"^(?P<variant>.+)_p(?P<prompt>\d+)_n(?P<gen>\d+)$")


def parse_summary(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_rates(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    text = path.read_text().strip()
    if not text:
        return None, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    pp = None
    tg = None
    for row in data:
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        avg_ts = float(row.get("avg_ts", 0.0))
        if n_prompt > 0 and n_gen == 0:
            pp = avg_ts
        if n_gen > 0:
            tg = avg_ts
    return pp, tg


def pct(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def failure_reason(summary: dict[str, str], has_rates: bool) -> str:
    if summary.get("limit_triggered") == "1":
        limit = summary.get("max_peak_mib", "")
        peak = summary.get("peak_mib", "")
        return f"MAX_PEAK_MIB exceeded (peak={peak}, limit={limit})"
    if not has_rates:
        return "missing or invalid llama-bench JSON"
    if summary.get("exit_code") not in ("", "0"):
        return f"exit_code={summary.get('exit_code')}"
    return ""


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        raise SystemExit(2)

    run_dir = Path(sys.argv[1])
    config = parse_summary(run_dir / "config.txt")
    base_model = Path(config["base_model"])
    oscar_model = Path(config["oscar_model"])
    prompt = int(config["prompt_tokens"])

    dims_by_model = {
        base_model.name: model_dims(read_metadata(base_model)),
        oscar_model.name: model_dims(read_metadata(oscar_model)),
    }

    rows = []
    for summary_path in sorted(run_dir.glob("*.summary.txt")):
        summary = parse_summary(summary_path)
        label = summary.get("label", summary_path.name.removesuffix(".summary.txt"))
        if label == "plain_int3_unsupported":
            rows.append({
                "variant": "plain_int3",
                "status": "unsupported",
                "reason": summary.get("reason", ""),
            })
            continue
        case_meta = run_dir / f"{label}.case.txt"
        if case_meta.exists():
            summary.update(parse_summary(case_meta))

        match = LABEL_RE.match(label)
        if not match:
            continue
        pp, tg = load_rates(run_dir / f"{label}.json")
        has_rates = pp is not None or tg is not None
        model_name = summary["model"]
        kv_type = summary["cache_type_k"]
        kv_mib = mib(estimate_bytes(dims_by_model[model_name], prompt, kv_type))
        rows.append({
            "variant": match.group("variant"),
            "status": "ok" if summary.get("exit_code") == "0" and has_rates else "failed",
            "model": model_name,
            "prompt": int(match.group("prompt")),
            "gen": int(match.group("gen")),
            "cache_k": summary["cache_type_k"],
            "cache_v": summary["cache_type_v"],
            "kv_pool_mib": kv_mib,
            "baseline_mib": int(summary.get("baseline_mib", 0)),
            "peak_mib": int(summary.get("peak_mib", 0)),
            "delta_mib": int(summary.get("delta_mib", 0)),
            "max_peak_mib": summary.get("max_peak_mib", ""),
            "limit_triggered": summary.get("limit_triggered", ""),
            "exit_code": summary.get("exit_code", ""),
            "pp_tps": pp,
            "tg_tps": tg,
            "reason": failure_reason(summary, has_rates),
        })

    variant_order = {
        "baseline_bf16": 0,
        "oscar_turbo2_streamk": 1,
        "turbo2_streamk": 2,
        "turbo2_default": 3,
        "turbo3_default": 4,
        "oscar_int2": 5,
        "plain_int2": 6,
        "oscar_int4": 7,
        "plain_int4": 8,
        "plain_int3": 9,
    }
    rows.sort(key=lambda r: (variant_order.get(r["variant"], 99), str(r["variant"])))

    csv_path = run_dir / "summary.csv"
    md_path = run_dir / "summary.md"
    fields = [
        "variant",
        "status",
        "model",
        "prompt",
        "gen",
        "cache_k",
        "cache_v",
        "kv_pool_mib",
        "baseline_mib",
        "peak_mib",
        "delta_mib",
        "max_peak_mib",
        "limit_triggered",
        "exit_code",
        "pp_tps",
        "tg_tps",
        "reason",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "| variant | status | KV | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        kv = f"{row.get('cache_k', '')}/{row.get('cache_v', '')}".strip("/")
        lines.append(
            f"| {row['variant']} | {row['status']} | {kv} | {pct(row.get('kv_pool_mib'))} | "
            f"{row.get('peak_mib', '')} | {row.get('delta_mib', '')} | {pct(row.get('pp_tps'))} | "
            f"{pct(row.get('tg_tps'))} | {row.get('reason', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
