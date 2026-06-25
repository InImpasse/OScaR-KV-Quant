#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def parse_label(label: str) -> tuple[str, str, str]:
    # pure_p512_oscar_int2 -> mode, prompt, variant
    mode, rest = label.split("_p", 1)
    prompt, variant = rest.split("_", 1)
    return mode, prompt, variant


def bench_metrics(path: Path) -> tuple[str, str, str]:
    data = json.loads(path.read_text())
    pp_tps = ""
    tg_tps = ""
    for row in data:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        tps = row.get("avg_ts")
        if tps is None:
            continue
        tps_s = f"{float(tps):.6f}"
        if n_prompt > 0 and n_gen == 0:
            pp_tps = tps_s
        elif n_gen > 0 and n_prompt == 0:
            tg_tps = tps_s
    status = "ok" if pp_tps else "failed"
    reason = "" if pp_tps else "missing pp row"
    return pp_tps, tg_tps, status, reason


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/mixed_vec_int2_ramp_current")
    rows: list[dict[str, str]] = []

    for summary in sorted(out_dir.glob("*.summary.txt")):
        label = summary.name.removesuffix(".summary.txt")
        meta = parse_summary(summary)
        mode, prompt, variant = parse_label(label)
        pp_tps = ""
        tg_tps = ""
        status = "failed"
        reason = meta.get("exit_code", "")
        json_path = out_dir / f"{label}.json"
        if json_path.is_file():
            try:
                pp_tps, tg_tps, status, json_reason = bench_metrics(json_path)
                if status != "ok":
                    reason = json_reason
            except json.JSONDecodeError:
                reason = "invalid json"
        rows.append({
            "label": label,
            "mode": mode,
            "prompt": prompt,
            "variant": variant,
            "status": status,
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "peak_mib": meta.get("peak_mib", ""),
            "reason": reason if status != "ok" else "",
        })

    csv_path = out_dir / "ramp.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Mixed vec INT2 ramp", "", f"out_dir={out_dir}", "", "| label | status | pp tok/s | peak MiB |", "|---|---|---:|---:|"]
    for row in rows:
        md_lines.append(f"| {row['label']} | {row['status']} | {row['pp_tps']} | {row['peak_mib']} |")
    (out_dir / "ramp.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
