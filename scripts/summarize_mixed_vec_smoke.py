#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/mixed_vec_smoke_current")
    require(out_dir.is_dir(), f"missing smoke out dir: {out_dir}")

    direct_summary = out_dir / "direct/summary.txt"
    require(direct_summary.is_file(), "missing direct completion summary")

    rows: list[dict[str, str]] = []
    for line in direct_summary.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        label, rest = line.split(" ", 1)
        rc = rest.split(" ", 1)[0].removeprefix("rc=")
        rows.append({"path": label, "rc": rc, "note": rest})

    quality_csv = out_dir / "quality/summary.csv"
    quality_rows: list[dict[str, str]] = []
    if quality_csv.is_file():
        quality_rows = list(csv.DictReader(quality_csv.open()))

    md_lines = [
        "# Mixed vec smoke summary",
        "",
        f"out_dir={out_dir}",
        "",
        "## Direct completion",
        "",
        "| path | rc |",
        "|---|---:|",
    ]
    for row in rows:
        md_lines.append(f"| {row['path']} | {row['rc']} |")

    if quality_rows:
        md_lines.extend(["", "## GPQA/GSM8K 3-case", "", "| variant | dataset | correct | total |", "|---|---|---:|---:|"])
        for row in quality_rows:
            md_lines.append(
                f"| {row['variant']} | {row['dataset']} | {row.get('correct', '')} | {row.get('total', '')} |"
            )

    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "key", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"section": "direct", "key": row["path"], "value": row["rc"]})
        for row in quality_rows:
            writer.writerow({
                "section": "quality",
                "key": f"{row['variant']}/{row['dataset']}",
                "value": f"{row.get('correct', '')}/{row.get('total', '')}",
            })

    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
