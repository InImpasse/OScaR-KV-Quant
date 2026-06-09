"""Validate OSCAR rotation calibration and optional GPQA accuracy parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _score(eval_dir: Path) -> float:
    metrics = _load_json(eval_dir / "metrics.json")
    return float(metrics["score"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check OSCAR rotation metadata against calibration/accuracy gates."
    )
    parser.add_argument("--rot-dir", type=Path, required=True)
    parser.add_argument("--min-calib-tokens", type=int, default=27000)
    parser.add_argument("--min-calib-prompts", type=int, default=1)
    parser.add_argument("--max-orthogonality-error", type=float, default=1e-4)
    parser.add_argument("--bf16-eval-dir", type=Path)
    parser.add_argument("--oscar-eval-dir", type=Path)
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.05,
        help="Allowed absolute score drop when eval dirs are provided.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta_path = args.rot_dir / "rotation_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}")

    meta = _load_json(meta_path)
    failures: list[str] = []

    if meta.get("method") != "qqt_sst":
        failures.append(f"method={meta.get('method')} expected qqt_sst")
    if meta.get("composition") != "r_h_pbr":
        failures.append(f"composition={meta.get('composition')} expected r_h_pbr")

    calib = meta.get("calibration") or {}
    tokens = int(calib.get("dumped_tokens", 0))
    prompts = int(calib.get("num_prompts_captured", 0))
    errors = int(calib.get("runner_error_prompts", 0))
    if tokens < args.min_calib_tokens:
        failures.append(f"calib tokens={tokens} < {args.min_calib_tokens}")
    if prompts < args.min_calib_prompts:
        failures.append(f"calib prompts={prompts} < {args.min_calib_prompts}")
    if errors:
        failures.append(f"calib runner errors={errors}")

    for target, info in (meta.get("rotation_files") or {}).items():
        err = float(info.get("max_orthogonality_error", 1.0))
        if err > args.max_orthogonality_error:
            failures.append(
                f"{target} orthogonality error={err:.3e} > "
                f"{args.max_orthogonality_error:.3e}"
            )

    if args.bf16_eval_dir or args.oscar_eval_dir:
        if not args.bf16_eval_dir or not args.oscar_eval_dir:
            failures.append("provide both --bf16-eval-dir and --oscar-eval-dir")
        else:
            bf16 = _score(args.bf16_eval_dir)
            oscar = _score(args.oscar_eval_dir)
            drop = bf16 - oscar
            if drop > args.max_accuracy_drop:
                failures.append(
                    f"accuracy drop={drop:.4f} > {args.max_accuracy_drop:.4f} "
                    f"(bf16={bf16:.4f}, oscar={oscar:.4f})"
                )
            print(
                f"accuracy: bf16={bf16:.4f} oscar={oscar:.4f} "
                f"drop={drop:.4f}"
            )

    print(
        f"rotation: tokens={tokens} prompts={prompts} method={meta.get('method')} "
        f"composition={meta.get('composition')}"
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
