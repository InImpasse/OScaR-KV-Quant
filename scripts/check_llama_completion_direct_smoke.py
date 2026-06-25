#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/llama_completion_direct_smoke_current"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_stdout(name: str) -> str:
    path = RUN / f"{name}.stdout.txt"
    require(path.is_file(), f"missing direct smoke stdout for {name}")
    return path.read_text(errors="replace")


def main() -> None:
    summary = RUN / "summary.txt"
    require(summary.is_file(), "missing llama-completion direct smoke summary")
    text = summary.read_text(errors="replace")
    require("oscar_bf16 rc=0" in text and "oscar_int4 rc=0" in text and "oscar_int2 rc=0" in text,
            "direct smoke must contain successful bf16/int4/int2 runs")

    bf16 = read_stdout("oscar_bf16")
    int4 = read_stdout("oscar_int4")
    int2 = read_stdout("oscar_int2")
    kq2_vbf16 = read_stdout("oscar_kq2_vbf16")
    kbf16_vq2 = read_stdout("oscar_kbf16_vq2")

    require("4" in bf16, "BF16 control should answer the direct arithmetic prompt with 4")
    require("2" in int4, "INT4 control should emit a normal digit on the direct arithmetic prompt")
    require("same>>" in int2, "exact q2/q2 direct smoke should preserve the observed malformed token pattern")
    require("2. 2 2" in kq2_vbf16, "K=q2,V=bf16 direct smoke should preserve the numeric-but-repetitive pattern")
    require("linese" in kbf16_vq2 or "·" in kbf16_vq2,
            "K=bf16,V=q2 direct smoke should preserve the observed malformed text pattern")

    print("llama-completion direct smoke checks passed")


if __name__ == "__main__":
    main()
