#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path("runs/no_hadamard_graph_gate_direct_smoke_current")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def read(name: str) -> str:
    path = ROOT / f"{name}.stdout.txt"
    require(path.exists(), f"missing {path}")
    return path.read_text().strip()


def main() -> None:
    require((ROOT / "summary.txt").exists(), f"missing {ROOT / 'summary.txt'}")

    bf16 = read("oscar_bf16")
    int4 = read("oscar_int4")
    int2 = read("oscar_int2")
    kq2 = read("oscar_kq2_vbf16")
    vq2 = read("oscar_kbf16_vq2")

    require("4" in bf16, "OSCAR BF16 direct prompt should still answer 4")
    require("4" in int4, "OSCAR INT4 direct prompt should recover 4 after graph no-Hadamard gate")
    require(int2 == "[end of text]", "OSCAR INT2 direct prompt remains incomplete and must not be called fixed")
    require(vq2 == "[end of text]", "V=q2 direct prompt remains incomplete and must not be called fixed")
    require("2" in kq2, "K=q2 direct prompt should remain degraded/repetitive in current triage")

    print("no-Hadamard graph gate direct smoke checks passed")


if __name__ == "__main__":
    main()
