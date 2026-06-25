#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "runs/llamacpp_32k_kv_matrix_current/combined.csv"
RAMP_PROMPTS = [512, 2048, 4096, 8192, 16384, 32768]
Q2_VARIANTS = ("plain_int2", "oscar_int2")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def has_speed(row: dict[str, str]) -> bool:
    return bool(row.get("pp_tps")) and bool(row.get("tg_tps"))


def q2_row(rows: list[dict[str, str]], variant: str, prompt: int) -> dict[str, str] | None:
    matches = [
        row for row in rows
        if row.get("variant") == variant and row.get("prompt") == str(prompt)
    ]
    return matches[0] if len(matches) == 1 else None


def step_status(rows: list[dict[str, str]], prompt: int) -> dict:
    variants = {}
    for variant in Q2_VARIANTS:
        row = q2_row(rows, variant, prompt)
        ok = row is not None and row.get("status") == "ok" and has_speed(row)
        variants[variant] = {
            "status": row.get("status", "missing") if row else "missing",
            "ok": ok,
            "pp_tps": row.get("pp_tps", "") if row else "",
            "tg_tps": row.get("tg_tps", "") if row else "",
            "reason": row.get("reason", "") if row else "",
            "run_dir": row.get("run_dir", "") if row else "",
        }

    any_ok = any(item["ok"] for item in variants.values())
    any_failed = any(item["status"] == "failed" for item in variants.values())
    return {
        "prompt": prompt,
        "any_ok": any_ok,
        "any_failed": any_failed,
        "variants": variants,
    }


def build_report(path: Path) -> dict:
    rows = load_rows(path)
    steps = [step_status(rows, prompt) for prompt in RAMP_PROMPTS]
    completed = [step["prompt"] for step in steps if step["any_ok"]]
    missing = [step["prompt"] for step in steps if not step["any_ok"]]
    failed_32k = any(
        row.get("variant") in Q2_VARIANTS
        and row.get("prompt") == "32768"
        and row.get("status") == "failed"
        for row in rows
    )

    next_prompt = missing[0] if missing else None
    if failed_32k:
        recommendation = "hold_32k_q2"
        command_hint = "Prefer code/profiler work before repeating the known 32k q2 NO-GO path."
    elif next_prompt is None:
        recommendation = "ramp_complete"
        command_hint = "All q2 ramp prompts have valid speed rows."
    else:
        recommendation = "run_next_prompt"
        command_hint = f"Run a single q2/int2 case at prompt {next_prompt} before any larger prompt."

    return {
        "matrix": str(path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path),
        "ramp_prompts": RAMP_PROMPTS,
        "completed_prompts": completed,
        "next_prompt": next_prompt,
        "failed_32k_q2": failed_32k,
        "recommendation": recommendation,
        "command_hint": command_hint,
        "steps": steps,
    }


def print_text(report: dict) -> None:
    print("# q2/int2 ramp gate")
    print()
    print(f"matrix={report['matrix']}")
    print("ramp_prompts=" + ",".join(str(p) for p in report["ramp_prompts"]))
    print("completed_prompts=" + ",".join(str(p) for p in report["completed_prompts"]))
    print(f"next_prompt={report['next_prompt'] if report['next_prompt'] is not None else ''}")
    print(f"failed_32k_q2={str(report['failed_32k_q2']).lower()}")
    print(f"recommendation={report['recommendation']}")
    print(f"command_hint={report['command_hint']}")
    print()
    print("## Steps")
    for step in report["steps"]:
        variant_notes = []
        for variant, data in step["variants"].items():
            note = data["status"]
            if data["pp_tps"]:
                note += f"/pp={data['pp_tps']}"
            if data["reason"]:
                note += f"/{data['reason']}"
            variant_notes.append(f"{variant}:{note}")
        print(f"- {step['prompt']}: any_ok={str(step['any_ok']).lower()} | " + "; ".join(variant_notes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the cautious q2/int2 ramp gate from archived llama.cpp results.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(args.matrix)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
