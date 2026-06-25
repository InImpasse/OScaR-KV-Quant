#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def one(rows: list[dict[str, str]], variant: str, prompt: str) -> dict[str, str] | None:
    matches = [r for r in rows if r.get("variant") == variant and r.get("prompt") == prompt]
    return matches[0] if len(matches) == 1 else None


def as_float(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key, "")
    return None if value == "" else float(value)


def status(ok: bool) -> str:
    return "complete" if ok else "incomplete"


def file_has(path: Path, needles: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"
    text = path.read_text(errors="replace")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        return False, "missing markers: " + ", ".join(missing)
    return True, "markers present"


def q2_cuda_path_archive_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path}"

    try:
        rows = load(path)
    except (OSError, csv.Error) as exc:
        return False, f"could not read {path}: {exc}"

    def find(path_name: str, type_name: str) -> dict[str, str] | None:
        return next((r for r in rows if r.get("path") == path_name and r.get("type") == type_name), None)

    q2_kq = find("KQ", "q2_0")
    q4_kq = find("KQ", "q4_0")
    q2_v = find("V", "q2_0")
    kq_dispatch = find("dispatch", "KQ")
    v_dispatch = find("dispatch", "V")

    checks = [
        (
            "q2_kq_exact_lut",
            q2_kq is not None
            and q2_kq.get("function") == "vec_dot_fattn_vec_KQ_q2_0_chunk"
            and q2_kq.get("dp4a_calls") == "3"
            and q2_kq.get("lut") == "sign+high"
            and q2_kq.get("mean_term") == "m*usum"
            and q2_kq.get("dispatch") == "yes",
        ),
        (
            "q4_kq_reference",
            q4_kq is not None
            and q4_kq.get("dp4a_calls") == "1"
            and q4_kq.get("dispatch") == "yes",
        ),
        (
            "q2_v_scalar_decode",
            q2_v is not None
            and q2_v.get("function") == "dequantize_V_q2_0"
            and q2_v.get("scalar_decode") == "yes"
            and q2_v.get("dispatch") == "yes",
        ),
        ("kq_dispatch", kq_dispatch is not None and kq_dispatch.get("dispatch") == "yes"),
        ("v_dispatch", v_dispatch is not None and v_dispatch.get("dispatch") == "yes"),
    ]

    missing = [name for name, ok in checks if not ok]
    if missing:
        return False, "missing path facts: " + ", ".join(missing)

    fingerprints = [r.get("fingerprint", "") for r in (q2_kq, q4_kq, q2_v, kq_dispatch, v_dispatch) if r]
    if any(len(fingerprint) != 64 for fingerprint in fingerprints):
        return False, "expected 64-char function fingerprints"

    return True, f"archived path facts present ({len(rows)} rows)"


def int4_cli_quality_ok(path: Path) -> tuple[bool, str]:
    raw = path / "raw"
    if not raw.is_dir():
        return False, f"missing {raw}"

    expected = {
        "baseline_bf16_gpqa.json": (3, 10),
        "baseline_bf16_gsm8k.json": (4, 10),
        "oscar_int4_gpqa.json": (4, 10),
        "oscar_int4_gsm8k.json": (4, 10),
    }
    notes = []
    for name, (want_correct, want_total) in expected.items():
        file_path = raw / name
        if not file_path.is_file():
            return False, f"missing {file_path}"
        data = json.loads(file_path.read_text())
        states = data.get("task_states", {})
        correct = int(states.get("correct", -1))
        total = int(states.get("total", -1))
        if correct != want_correct or total != want_total:
            return False, f"{name}: expected {want_correct}/{want_total}, got {correct}/{total}"
        notes.append(f"{name.removesuffix('.json')}={correct}/{total}")

    return True, "; ".join(notes)


def write_reports(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "goal_status.csv"
    md_path = out_dir / "goal_status.md"
    fields = ["item", "status", "note"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| item | status | note |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['note']} |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current progress toward the 32k llama.cpp KV goal.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("runs/llamacpp_32k_kv_matrix_current/combined.csv"),
    )
    parser.add_argument(
        "--graph-ab",
        type=Path,
        default=Path("runs/cuda_graph_ab_512_current/graph_ab.csv"),
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    rows = load(args.matrix)
    graph_rows = load(args.graph_ab)

    bf16 = one(rows, "baseline_bf16", "32768")
    oscar_int4 = one(rows, "oscar_int4", "32768")
    plain_int4 = one(rows, "plain_int4", "32768")
    oscar_turbo2_streamk_32k = one(rows, "oscar_turbo2_streamk", "32768")
    turbo2_streamk_32k = one(rows, "turbo2_streamk", "32768")
    oscar_int2_32k = one(rows, "oscar_int2", "32768")
    oscar_int2_16k = one(rows, "oscar_int2", "16384")
    plain_int2_16k = one(rows, "plain_int2", "16384")
    graph_on = next((r for r in graph_rows if r.get("mode") == "on" and r.get("opt") == "1"), None)

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "32k_bf16_baseline",
        bf16 is not None and bf16.get("status") == "ok" and as_float(bf16, "pp_tps") is not None,
        f"pp={bf16.get('pp_tps', '') if bf16 else ''}, peak={bf16.get('peak_mib', '') if bf16 else ''}",
    ))

    int4_ok = True
    int4_notes = []
    for name, row in (("oscar_int4", oscar_int4), ("plain_int4", plain_int4)):
        peak_saved = as_float(row, "peak_saved_vs_bf16_mib")
        kv_saved = as_float(row, "kv_saved_vs_bf16_mib")
        ok = (
            row is not None
            and row.get("status") == "ok"
            and row.get("kv") == "q4_0/q4_0"
            and peak_saved is not None
            and kv_saved is not None
            and abs(peak_saved - kv_saved) <= 64.0
            and as_float(row, "pp_tps") is not None
        )
        int4_ok = int4_ok and ok
        int4_notes.append(f"{name}:pp={row.get('pp_tps', '') if row else ''},peak_saved={peak_saved},kv_saved={kv_saved}")
    checks.append(("32k_int4_memory_and_speed", int4_ok, "; ".join(int4_notes)))

    int4_quality_ok, int4_quality_note = int4_cli_quality_ok(
        Path("runs/oscar_int4_cli_quality_10_current")
    )
    checks.append(("int4_cli_quality_smoke", int4_quality_ok, int4_quality_note))

    int2_16k_ok = (
        oscar_int2_16k is not None
        and plain_int2_16k is not None
        and oscar_int2_16k.get("status") == "ok"
        and plain_int2_16k.get("status") == "ok"
        and as_float(oscar_int2_16k, "pp_tps") is not None
        and as_float(plain_int2_16k, "pp_tps") is not None
    )
    checks.append((
        "16k_int2_gate",
        int2_16k_ok,
        f"oscar_pp={oscar_int2_16k.get('pp_tps', '') if oscar_int2_16k else ''}, plain_pp={plain_int2_16k.get('pp_tps', '') if plain_int2_16k else ''}",
    ))

    turbo2_pp = as_float(oscar_turbo2_streamk_32k, "pp_tps")
    turbo2_peak = as_float(oscar_turbo2_streamk_32k, "peak_mib")
    turbo2_peak_saved = as_float(oscar_turbo2_streamk_32k, "peak_saved_vs_bf16_mib")
    turbo2_kv_saved = as_float(oscar_turbo2_streamk_32k, "kv_saved_vs_bf16_mib")
    int4_pp = as_float(oscar_int4, "pp_tps")
    int4_peak = as_float(oscar_int4, "peak_mib")
    bf16_pp = as_float(bf16, "pp_tps")
    bf16_peak = as_float(bf16, "peak_mib")
    turbo2_reference_ok = (
        oscar_turbo2_streamk_32k is not None
        and oscar_turbo2_streamk_32k.get("status") == "ok"
        and oscar_turbo2_streamk_32k.get("kv") == "turbo2/turbo2"
        and turbo2_pp is not None
        and int4_pp is not None
        and bf16_pp is not None
        and turbo2_pp > int4_pp
        and turbo2_pp > bf16_pp
        and turbo2_peak is not None
        and int4_peak is not None
        and bf16_peak is not None
        and turbo2_peak < int4_peak
        and turbo2_peak < bf16_peak
        and turbo2_peak_saved is not None
        and turbo2_kv_saved is not None
        and abs(turbo2_peak_saved - turbo2_kv_saved) <= 256.0
    )
    checks.append((
        "32k_turbo2_reference",
        turbo2_reference_ok,
        (
            f"oscar_turbo2_status={oscar_turbo2_streamk_32k.get('status', '') if oscar_turbo2_streamk_32k else ''},"
            f"oscar_turbo2_pp={turbo2_pp},oscar_turbo2_peak={turbo2_peak},"
            "note=Turbo2 is not exact OSCAR INT2"
        ),
    ))

    exact_q2_pp = as_float(oscar_int2_32k, "pp_tps")
    exact_q2_peak = as_float(oscar_int2_32k, "peak_mib")
    exact_q2_peak_saved = as_float(oscar_int2_32k, "peak_saved_vs_bf16_mib")
    exact_q2_kv_saved = as_float(oscar_int2_32k, "kv_saved_vs_bf16_mib")
    int2_32k_ok = (
        oscar_int2_32k is not None
        and oscar_int2_32k.get("status") == "ok"
        and oscar_int2_32k.get("kv") == "q2_0/q2_0"
        and exact_q2_pp is not None
        and int4_pp is not None
        and bf16_pp is not None
        and exact_q2_pp > int4_pp
        and exact_q2_pp > bf16_pp
        and exact_q2_peak is not None
        and int4_peak is not None
        and bf16_peak is not None
        and exact_q2_peak < int4_peak
        and exact_q2_peak < bf16_peak
        and exact_q2_peak_saved is not None
        and exact_q2_kv_saved is not None
        and abs(exact_q2_peak_saved - exact_q2_kv_saved) <= 256.0
    )
    checks.append((
        "32k_int2_speed_target",
        int2_32k_ok,
        (
            f"exact_q2_status={oscar_int2_32k.get('status', '') if oscar_int2_32k else ''},"
            f"exact_q2_kv={oscar_int2_32k.get('kv', '') if oscar_int2_32k else ''},"
            f"exact_q2_pp={exact_q2_pp},exact_q2_peak={exact_q2_peak},"
            f"oscar_int4_pp={int4_pp},oscar_int4_peak={int4_peak},"
            f"bf16_pp={bf16_pp},bf16_peak={bf16_peak}; "
            f"turbo2_reference_pp={turbo2_pp}"
        ),
    ))

    graph_no_help = (
        graph_on is not None
        and as_float(graph_on, "pp_pct_vs_off") is not None
        and as_float(graph_on, "pp_pct_vs_off") <= 0.0
    )
    checks.append((
        "cuda_graph_512_ab",
        graph_no_help,
        f"graph_on_pp_pct_vs_off={graph_on.get('pp_pct_vs_off', '') if graph_on else ''}",
    ))

    llama_only_ok, llama_only_note = file_has(Path("scripts/check_llamacpp_only.py"), [
        "third_party/OSCAR",
        "In" + "Impasse",
        "Runtime" + "Endpoint",
        "server" + "_args",
    ])
    checks.append(("llamacpp_only_guardrails", llama_only_ok, llama_only_note))

    entrypoint_ok, entrypoint_note = file_has(Path("scripts/check_execution_entrypoints.py"), [
        "DRY_RUN=\\\"${DRY_RUN:-1}\\\"",
        "RUN_REAL=\\\"${RUN_REAL:-0}\\\"",
        "CHECK_GPU_SNAPSHOT=\\\"${CHECK_GPU_SNAPSHOT:-0}\\\"",
        "ACK_RUN_LLAMA",
        "ACK_HEAVY_CONTEXT",
        "ACK_MATRIX_BENCH",
        "ACK_PPL_MATRIX",
    ])
    legacy_ok, legacy_note = file_has(Path("scripts/check_legacy_bench_safety.py"), [
        "run_llamacpp.sh",
        "bench_32k_llamacpp_kv.sh",
        "run_kv_ppl_matrix.sh",
        "ACK_Q2_RAMP_GATE_HOLD",
    ])
    checks.append((
        "execution_safety_guardrails",
        entrypoint_ok and legacy_ok,
        f"entrypoints={entrypoint_note}; legacy={legacy_note}",
    ))

    no_gpu_verifier_ok, no_gpu_verifier_note = file_has(Path("scripts/check_no_gpu_verifier_no_nvidia_smi.py"), [
        "fake = bin_dir / \"nvidia-smi\"",
        "CHECK_GPU_SNAPSHOT",
        "SKIP_NO_GPU_VERIFIER_SELFTEST",
        "default verifier should not call outer nvidia-smi",
    ])
    q2_profile_snapshot_ok, q2_profile_snapshot_note = file_has(Path("scripts/check_q2_profile_safety.py"), [
        "Q2_PROFILE_GPU_SNAPSHOT",
        "skip nvidia-smi by default",
    ])
    checks.append((
        "no_gpu_verifier_guard",
        no_gpu_verifier_ok and q2_profile_snapshot_ok,
        f"verifier={no_gpu_verifier_note}; q2_profile={q2_profile_snapshot_note}",
    ))

    command_helper_ok, command_helper_note = file_has(Path("scripts/check_32k_matrix_commands.py"), [
        "ACK_HEAVY_32K=1",
        "ACK_Q2_32K_NOGO=1",
        "MAX_PEAK_MIB=7000",
        "POST_CASE_COOLDOWN_SEC",
        "ack-real",
        "ack-32k-q2-real",
        "ack-q2-ramp-gate-hold",
        "ACK_Q2_RAMP_GATE_HOLD=1",
        "require_no_output_dirs",
        "plain_int3",
    ])
    q2_ramp_ok, q2_ramp_note = file_has(Path("scripts/check_32k_ramp_commands.py"), [
        "PROMPT_TOKENS=512",
        "PROMPT_TOKENS=2048",
        "PROMPT_TOKENS=4096",
        "PROMPT_TOKENS=8192",
        "PROMPT_TOKENS=16384",
        "PROMPT_TOKENS=32768",
        "CASE_TIMEOUT_SEC=240",
        "ACK_Q2_32K_NOGO=1",
        "MAX_PEAK_MIB=7000",
        "POST_CASE_COOLDOWN_SEC",
        "ack-real",
        "ack-32k-q2-real",
        "ack-q2-ramp-gate-hold",
        "ACK_Q2_RAMP_GATE_HOLD=1",
        "require_no_output_dirs",
    ])
    checks.append((
        "recovery_command_guardrails",
        command_helper_ok and q2_ramp_ok,
        f"matrix={command_helper_note}; q2_ramp={q2_ramp_note}",
    ))

    cooldown_ok, cooldown_note = file_has(Path("scripts/check_post_case_cooldown.py"), [
        "fake-llama-bench",
        "nvidia-smi",
        "POST_CASE_COOLDOWN_SEC",
        "returncode == 125",
        "GPU did not cool down",
        "baseline_bf16_p512_n1.summary.txt",
    ])
    checks.append(("post_case_cooldown_guard", cooldown_ok, cooldown_note))

    readiness_ok, readiness_note = file_has(Path("scripts/check_recovery_readiness.py"), [
        "overall_status=complete",
        "exact_int2_research_status=incomplete",
        "can_mark_complete=true",
        "Turbo2 is a separate reference",
        "gpu_snapshot=",
        "archive_checksums_ok",
        "Archive checksums",
        "matrix",
        "cuda_graph_ab",
        "goal_status",
        "--json",
        "--no-gpu",
        "can_mark_complete",
        "32k_int2_speed_target",
        "q2_cuda_path_archive_fresh",
        "q2_kq_dp4a_calls",
        "q2_ramp_recommendation=hold_32k_q2",
        "q2_ramp_next_prompt=512",
        "Q2 ramp gate",
        "recommended_next_actions",
        "512/2k/4k/8k/16k q2 ramp",
        "--real --ack-real",
        "--ack-32k-q2-real",
        "--ack-q2-ramp-gate-hold",
    ])
    checks.append(("recovery_readiness_report", readiness_ok, readiness_note))

    q2_ramp_gate_ok, q2_ramp_gate_note = file_has(Path("scripts/check_q2_ramp_gate.py"), [
        "ramp_prompts=512,2048,4096,8192,16384,32768",
        "completed_prompts=16384",
        "failed_32k_q2=true",
        "recommendation=hold_32k_q2",
        "next_prompt",
        "subprocess",
    ])
    checks.append(("q2_ramp_gate_guard", q2_ramp_gate_ok, q2_ramp_gate_note))

    q2_ramp_harness_ok, q2_ramp_harness_note = file_has(Path("scripts/check_q2_ramp_gate_harness_guard.py"), [
        "ACK_Q2_RAMP_GATE_HOLD",
        "hold_32k_q2",
        "fake llama-bench should not run",
        "nvidia-smi should not run",
        "normal GPU baseline guard",
    ])
    checks.append(("q2_ramp_gate_harness_guard", q2_ramp_harness_ok, q2_ramp_harness_note))

    futuremls_plan_ok, futuremls_plan_note = file_has(Path("scripts/check_futuremls_q2_cuda_plan.py"), [
        "futuremls/zhongzhu/llamacpp",
        "kernel_flash_attn_mixed_mm_q2_0_f16_d128",
        "mm_mixed_pass",
        "Q=8",
        "C=64",
        "per-KV scalar path",
        "LLAMA_TURBO_VEC_STREAM_K",
        "stream-k launch",
    ])
    futuremls_doc_ok, futuremls_doc_note = file_has(Path("docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md"), [
        "dedicated D=128 q2/q2 prefill kernel",
        "Do not spend more effort on small changes inside the current q2 vec dot path",
        "run_q2_ramp_next.py",
        "ACK_Q2_RAMP_GATE_HOLD=1",
    ])
    checks.append((
        "futuremls_q2_cuda_port_plan",
        futuremls_plan_ok and futuremls_doc_ok,
        f"checker={futuremls_plan_note}; doc={futuremls_doc_note}",
    ))

    q2_static_ok, q2_static_note = file_has(Path("scripts/check_q2_cuda_static.py"), [
        "q2_chunk_dp4a",
        "q4_kq_dp4a",
        "m*usum",
        "Q2_0_FATTN_SIGN_LUT",
        "Q2_0_FATTN_HIGH_LUT",
    ])
    checks.append(("q2_cuda_static_guardrails", q2_static_ok, q2_static_note))

    q2_archive_ok, q2_archive_note = q2_cuda_path_archive_ok(
        Path("runs/q2_cuda_path_current/q2_cuda_path.csv")
    )
    checks.append(("q2_cuda_path_archive", q2_archive_ok, q2_archive_note))

    research_only = {
        "16k_int2_gate",
        "32k_turbo2_reference",
        "32k_int2_speed_target",
        "cuda_graph_512_ab",
        "futuremls_q2_cuda_port_plan",
        "q2_cuda_path_archive",
    }
    deliverable_complete = all(ok for name, ok, _ in checks if name not in research_only)
    exact_int2_complete = int2_32k_ok

    report_rows = [{
        "item": "overall_status",
        "status": status(deliverable_complete),
        "note": "current deliverable is oscar_int4; exact INT2 remains a separate incomplete research target",
    }, {
        "item": "exact_int2_research_status",
        "status": status(exact_int2_complete),
        "note": "requires valid 32k q2_0/q2_0 speed/memory before exact INT2 can be called complete",
    }]
    print(f"overall_status={status(deliverable_complete)}")
    print(f"exact_int2_research_status={status(exact_int2_complete)}")
    for name, ok, note in checks:
        row = {"item": name, "status": status(ok), "note": note}
        report_rows.append(row)
        print(f"{name}={row['status']} | {note}")

    if args.out_dir is not None:
        write_reports(report_rows, args.out_dir)


if __name__ == "__main__":
    main()
