#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md"

Q2_8K_BASELINE = 310.0
BF16_32K_PP = 2486.4
TILED_RECOMMEND_PP = 700.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_ramp(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open()))


def best_pp(rows: list[dict[str, str]], *, prompt: str, mode: str | None = None) -> float | None:
    best = None
    for row in rows:
        if row.get("prompt") != prompt and f"_p{prompt}_" not in row.get("label", ""):
            continue
        if mode and row.get("mode") != mode:
            continue
        if row.get("status") != "ok":
            continue
        try:
            value = float(row["pp_tps"])
        except (KeyError, ValueError):
            continue
        best = value if best is None else max(best, value)
    return best


def main() -> None:
    ramp_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/mixed_vec_int2_ramp_current")
    rows = load_ramp(ramp_dir / "ramp.csv")

    plan = PLAN.read_text()
    require("kernel_flash_attn_mixed_mm_q2_0_f16_d128" in plan, "FutureMLS plan must document tiled kernel")
    require("Do not spend more effort on small changes inside the current q2 vec dot path" in plan,
            "FutureMLS plan must warn against vec dot micro-patches")

    pure_8k = best_pp(rows, prompt="8192", mode="pure")
    mixed_vec_8k = best_pp(rows, prompt="8192", mode="mixed_vec")
    mixed_fused_8k = best_pp(rows, prompt="8192", mode="mixed_fused")

    recommendation = "continue_vec_convergence"
    reasons: list[str] = []

    if mixed_vec_8k is None and mixed_fused_8k is None and pure_8k is None:
        recommendation = "insufficient_data"
        reasons.append("no valid 8k ramp rows")
    else:
        best = max(v for v in (pure_8k, mixed_vec_8k, mixed_fused_8k) if v is not None)
        if best < Q2_8K_BASELINE:
            recommendation = "hold_changes"
            reasons.append(f"8k best {best:.1f} tok/s below q2 baseline gate {Q2_8K_BASELINE:.1f}")
        elif best < TILED_RECOMMEND_PP:
            recommendation = "pivot_to_tiled_mixed_fa"
            reasons.append(
                f"8k best {best:.1f} tok/s clears q2 gate but is far below tiled target {TILED_RECOMMEND_PP:.1f}"
            )
            reasons.append(f"bf16 32k reference {BF16_32K_PP:.1f} tok/s still out of reach for vec-only path")
        else:
            recommendation = "continue_vec_convergence"
            reasons.append(f"8k best {best:.1f} tok/s is healthy enough to keep vec/tiled hybrid work")

    out = ramp_dir / "tiled_decision.md"
    lines = [
        "# Mixed vec tiled-kernel decision",
        "",
        f"ramp_dir={ramp_dir}",
        f"recommendation={recommendation}",
        "",
        "## 8k observations",
        f"- pure_8k_pp={pure_8k if pure_8k is not None else 'missing'}",
        f"- mixed_fused_8k_pp={mixed_fused_8k if mixed_fused_8k is not None else 'missing'}",
        f"- mixed_vec_8k_pp={mixed_vec_8k if mixed_vec_8k is not None else 'missing'}",
        "",
        "## Reasons",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend([
        "",
        "## Next step",
        "- pivot_to_tiled_mixed_fa: implement FutureMLS-style Q=8/C=64 mixed prefill per docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md",
        "- continue_vec_convergence: keep HP/LP V vec alignment and re-run ramp",
        "- hold_changes: do not promote kernel changes until 8k regresses are understood",
        "- insufficient_data: run scripts/run_mixed_vec_int2_ramp.sh with ACK_MIXED_VEC_RAMP=1 DRY_RUN=0",
    ])
    out.write_text("\n".join(lines) + "\n")
    print(f"recommendation={recommendation}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
