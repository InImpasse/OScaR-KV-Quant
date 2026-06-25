#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    graph = (ROOT / "third_party/OSCAR/src/llama-graph.cpp").read_text()
    report = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()

    require("prompt processing keeps filling HP but uses the normal q2 FA path" in graph,
            "llama graph must document default prompt prefill HP attention behavior")
    require("ubatch.n_tokens <= 2*ubatch.n_seqs_unq" in graph,
            "HP attention should remain generation-batch gated by default")
    require("LLAMA_KV_HP_PREFILL_ATTENTION" in graph,
            "HP prefill attention experiment must be explicitly env-gated")
    require("runs/q2_hp_prefill_attention_probe_current" in report,
            "triage report must include the HP prefill attention probe")
    require("HP prefill" in report and "not enough to recover exact" in report,
            "triage report must record that HP prefill attention did not rescue q2")
    require("runs/q2_hp_fullctx_probe_current" in report,
            "triage report must include full-context HP probe")
    require("HP recent cannot be" in report and "treated as a q2 prefill quality fix" in report,
            "triage report must state HP recent is not a q2 prefill fix")

    print("HP prefill q2 limit checks passed")


if __name__ == "__main__":
    main()
