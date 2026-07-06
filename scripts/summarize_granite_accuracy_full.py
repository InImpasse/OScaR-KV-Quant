#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path


VARIANTS = ("baseline_bf16", "oscar_int4", "plain_int4")
VARIANT_LABEL = {
    "baseline_bf16": "BF16",
    "oscar_int4": "OSCAR INT4",
    "plain_int4": "Plain INT4",
}
DATASET_LABEL = {
    "gpqa": ("GPQA", "Score"),
    "gsm8k": ("GSM8K", "Accuracy"),
    "math500": ("MATH500", "Score"),
    "humaneval": ("HumanEval", "Pass@1"),
    "aime2025": ("AIME25", "Score"),
    "lcb_v6": ("LCB V6", "Pass@1"),
}


def fmt_score(value):
    return "" if value is None else f"{value:.2f}"


def fmt_delta(value, base):
    if value is None or base is None:
        return ""
    delta = value - base
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} pt"


def load_non_lcb(run_dir: Path):
    rows = {}
    summary = run_dir / "non_lcb" / "summary.csv"
    if not summary.exists():
        return rows
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            score = row.get("score_pct")
            rows[(row["variant"], row["dataset"])] = None if score == "" else float(score)
    return rows


def find_lcb_score(path: Path):
    text = path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        try:
            value = float(lines[-1])
            if 0.0 <= value <= 1.0:
                return value * 100.0
        except ValueError:
            pass
    patterns = [
        r"pass@1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        r"pass_at_1[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        r'"pass@1"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r'"pass_at_1"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value = float(matches[-1])
            return value * 100.0 if value <= 1.0 else value
    return None


def load_lcb(run_dir: Path):
    rows = {}
    lcb_dir = run_dir / "lcb_v6"
    if not lcb_dir.exists():
        return rows
    for variant in VARIANTS:
        candidates = []
        log = lcb_dir / "logs" / f"{variant}.lcb.log"
        if log.exists():
            candidates.append(log)
        out_dir = lcb_dir / "raw" / variant / "lcb_output"
        if out_dir.exists():
            candidates.extend(sorted(out_dir.rglob("*.json")))
            candidates.extend(sorted(out_dir.rglob("*.txt")))
        for candidate in candidates:
            score = find_lcb_score(candidate)
            if score is not None:
                rows[(variant, "lcb_v6")] = score
                break
    return rows


def load_humaneval_passk(run_dir: Path):
    rows = {}
    raw_dir = run_dir / "non_lcb" / "raw"
    if not raw_dir.exists():
        return rows
    for variant in VARIANTS:
        path = raw_dir / f"{variant}_humaneval.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        cases = data.get("task_states", {}).get("cases", {})
        completed = [c for c in cases.values() if c.get("status") == "ok"]
        if not completed:
            continue
        by_problem = {}
        for case in completed:
            by_problem.setdefault(case.get("problem_idx"), []).append(case)
        for problem_cases in by_problem.values():
            problem_cases.sort(key=lambda c: (int(c.get("chunk_idx", 0)), str(c.get("task_id", ""))))
        for k in (1, 2, 5):
            usable = [cs for cs in by_problem.values() if len(cs) >= k]
            if not usable:
                continue
            passed = sum(1 for cs in usable if any(c.get("correct") for c in cs[:k]))
            rows[(variant, f"humaneval_pass{k}")] = 100.0 * passed / len(usable)
    return rows


def humaneval_passk_redundant(scores: dict) -> bool:
    """True when Pass@2/5 add no information (typically temperature=0)."""
    for variant in VARIANTS:
        pass1 = scores.get((variant, "humaneval_pass1"))
        if pass1 is None:
            continue
        for k in (2, 5):
            other = scores.get((variant, f"humaneval_pass{k}"))
            if other is not None and round(pass1, 4) != round(other, 4):
                return False
    return True


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    scores = {}
    scores.update(load_non_lcb(run_dir))
    scores.update(load_lcb(run_dir))
    scores.update(load_humaneval_passk(run_dir))

    lines = [
        "| Benchmark | Metric | BF16 | OSCAR INT4 | Delta vs BF16 | Plain INT4 | Delta vs BF16 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = [
        ("gpqa", "GPQA", "Score"),
        ("gsm8k", "GSM8K", "Accuracy"),
        ("math500", "MATH500", "Score"),
        ("lcb_v6", "LCB V6", "Pass@1"),
        ("humaneval_pass1", "HumanEval", "Pass@1"),
    ]
    if not humaneval_passk_redundant(scores):
        ordered.extend(
            [
                ("humaneval_pass2", "HumanEval", "Pass@2"),
                ("humaneval_pass5", "HumanEval", "Pass@5"),
            ]
        )
    ordered.append(("aime2025", "AIME25", "Score"))
    for dataset, label, metric in ordered:
        bf16 = scores.get(("baseline_bf16", dataset))
        oscar = scores.get(("oscar_int4", dataset))
        plain = scores.get(("plain_int4", dataset))
        lines.append(
            f"| {label} | {metric} | {fmt_score(bf16)} | {fmt_score(oscar)} | "
            f"{fmt_delta(oscar, bf16)} | {fmt_score(plain)} | {fmt_delta(plain, bf16)} |"
        )

    lines.extend([
        "",
        "Notes:",
        "- Non-LCB results are parsed from `non_lcb/summary.csv`.",
        "- LCB parsing is best-effort because LiveCodeBench output filenames vary by version.",
        "- HumanEval Pass@1/2/5 use 164 problems x `HUMANEVAL_SAMPLES` (default 5), matching the SGLang suite shape.",
        "- HumanEval sampling defaults to `HUMANEVAL_TEMPERATURE=0.2`; Pass@2/5 only differ when sampling is enabled.",
        "- AIME25 defaults to `AIME25_N_PREDICT=4096`; answers require an explicit `\\boxed{}` after thinking blocks are stripped.",
    ])

    out = run_dir / "accuracy_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
