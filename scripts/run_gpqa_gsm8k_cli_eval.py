#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_PY = ROOT / "third_party/OSCAR/examples/llama-eval/llama-eval.py"


def oscar_env(turbo_stream_k: str = "0") -> dict[str, str]:
    return {
        "LLAMA_KV_NO_HADAMARD": "1",
        "LLAMA_KV_CLIP_RATIO": "0",
        "LLAMA_KV_CLIP_RATIO_K": "0",
        "LLAMA_KV_CLIP_RATIO_V": "0",
        "LLAMA_TURBO_VEC_STREAM_K": turbo_stream_k,
    }


def oscar_q2_env(turbo_stream_k: str = "0") -> dict[str, str]:
    env = oscar_env(turbo_stream_k)
    env["LLAMA_KV_Q2_0_OWHT"] = "1"
    return env


VARIANTS = {
    "baseline_bf16": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "bf16",
        "cache_v": "bf16",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "oscar_bf16": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "bf16",
        "cache_v": "bf16",
        "env": oscar_env(),
    },
    "oscar_turbo2_streamk": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "turbo2",
        "cache_v": "turbo2",
        "env": oscar_env("1"),
    },
    "turbo2_streamk": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "turbo2",
        "cache_v": "turbo2",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "1"},
    },
    "oscar_turbo3": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "turbo3",
        "cache_v": "turbo3",
        "env": oscar_env(),
    },
    "plain_turbo3": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "turbo3",
        "cache_v": "turbo3",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "plain_int3": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "turbo3",
        "cache_v": "turbo3",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "plain_int2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q2_0",
        "cache_v": "q2_0",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "plain_int2_nohad": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q2_0",
        "cache_v": "q2_0",
        "env": {"LLAMA_KV_NO_HADAMARD": "1", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "plain_kq2_vbf16": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q2_0",
        "cache_v": "bf16",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "plain_kbf16_vq2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "bf16",
        "cache_v": "q2_0",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "oscar_int2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q2_0",
        "cache_v": "q2_0",
        "env": oscar_q2_env(),
    },
    "oscar_int2_mixed": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q2_0",
        "cache_v": "q2_0",
        "env": {
            **oscar_q2_env(),
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "256",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
        },
    },
    "oscar_int2_mixed_4k": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q2_0",
        "cache_v": "q2_0",
        "env": {
            **oscar_q2_env(),
            "LLAMA_KV_HP_SINK": "0",
            "LLAMA_KV_HP_RECENT": "4096",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
        },
    },
    "oscar2_int2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": oscar_env(),
    },
    "oscar2_int2_mixed": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "256",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
        },
    },
    "oscar2_int2_mixed_4k": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_HP_SINK": "0",
            "LLAMA_KV_HP_RECENT": "4096",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
        },
    },
    "oscar2_int2_mixed_vec": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "256",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
            "LLAMA_KV_MIXED_VEC_MAIN": "1",
        },
    },
    "oscar_int2_current": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_MIXED_VEC_RAW": "1",
            "LLAMA_KV_MIXED_VEC_MAIN": "1",
            "LLAMA_KV_OSCAR2_ALLOW_STAGED_FA": "1",
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "256",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
            "LLAMA_KV_HP_STAGED_COMBINE": "1",
            "LLAMA_KV_HP_STAGED_MASK_SKIP": "0",
        },
    },
    "oscar_int2_current_recent128": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_MIXED_VEC_RAW": "1",
            "LLAMA_KV_MIXED_VEC_MAIN": "1",
            "LLAMA_KV_OSCAR2_ALLOW_STAGED_FA": "1",
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "128",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
            "LLAMA_KV_HP_STAGED_COMBINE": "1",
            "LLAMA_KV_HP_STAGED_MASK_SKIP": "0",
        },
    },
    "oscar_int2_current_recent64": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_MIXED_VEC_RAW": "1",
            "LLAMA_KV_MIXED_VEC_MAIN": "1",
            "LLAMA_KV_OSCAR2_ALLOW_STAGED_FA": "1",
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "64",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
            "LLAMA_KV_HP_STAGED_COMBINE": "1",
            "LLAMA_KV_HP_STAGED_MASK_SKIP": "0",
        },
    },
    "oscar2_int2_mixed_vec_twotier": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "oscar2",
        "cache_v": "oscar2",
        "env": {
            **oscar_env(),
            "LLAMA_KV_HP_SINK": "64",
            "LLAMA_KV_HP_RECENT": "256",
            "LLAMA_KV_HP_PREFILL_ATTENTION": "1",
            "LLAMA_KV_MIXED_VEC_RAW": "1",
            "LLAMA_KV_MIXED_VEC_MAIN": "1",
            "LLAMA_KV_MIXED_VEC_LP_TWO_TIER_STRIDE": "4",
            "LLAMA_KV_MIXED_VEC_LP_TWO_TIER_TAIL": "1024",
            "LLAMA_KV_MIXED_VEC_LP_TWO_TIER_MODE": "end",
            "LLAMA_KV_MIXED_VEC_LP_TWO_TIER_WEIGHTED": "1",
        },
    },
    "oscar_kq2_vbf16": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q2_0",
        "cache_v": "bf16",
        "env": oscar_q2_env(),
    },
    "oscar_kbf16_vq2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "bf16",
        "cache_v": "q2_0",
        "env": oscar_q2_env(),
    },
    "oscar_kq4_vq2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q4_0",
        "cache_v": "q2_0",
        "env": oscar_q2_env(),
    },
    "oscar_kq4_voscar2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q4_0",
        "cache_v": "oscar2",
        "env": oscar_env(),
    },
    "oscar_kq4_vbf16": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q4_0",
        "cache_v": "bf16",
        "env": oscar_env(),
    },
    "oscar_k_turbo3_v_int2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "turbo3",
        "cache_v": "q2_0",
        "env": oscar_q2_env(),
    },
    "plain_kq4_vq2": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q4_0",
        "cache_v": "q2_0",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "oscar_kq4_vturbo3": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q4_0",
        "cache_v": "turbo3",
        "env": oscar_env(),
    },
    "plain_kq4_vturbo3": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q4_0",
        "cache_v": "turbo3",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
    "oscar_int4": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf",
        "cache_k": "q4_0",
        "cache_v": "q4_0",
        "env": oscar_env(),
    },
    "plain_int4": {
        "model": "checkpoints/gguf/granite-4.0-1b-base-bf16.gguf",
        "cache_k": "q4_0",
        "cache_v": "q4_0",
        "env": {"LLAMA_KV_NO_HADAMARD": "0", "LLAMA_KV_CLIP_RATIO": "0", "LLAMA_TURBO_VEC_STREAM_K": "0"},
    },
}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("llama_eval", EVAL_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pick_indices(size: int, n_cases: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(size))
    rng.shuffle(indices)
    return indices[: min(n_cases, size)]


def clean_completion(stdout: str, prompt: str) -> str:
    idx = stdout.rfind(prompt)
    if idx >= 0:
        return stdout[:idx] + stdout[idx + len(prompt):]
    return stdout


def run_cli(llama_cli: Path, variant: dict, prompt: str, n_predict: int, args) -> tuple[str, int, float]:
    env = os.environ.copy()
    env.update({
        "LLAMA_KV_HP_SINK": "0",
        "LLAMA_KV_HP_RECENT": "0",
        "LLAMA_KV_HP_PREFILL_ATTENTION": "0",
        "LLAMA_KV_Q2_0_OWHT": "0",
    })
    env.update(variant["env"])
    if args.extra_env:
        for item in args.extra_env.split(","):
            if not item:
                continue
            key, _, value = item.partition("=")
            env[key] = value

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(prompt)
        prompt_file = Path(f.name)

    cmd = [
        str(llama_cli),
        "-m", str(ROOT / variant["model"]),
        "-c", str(args.ctx_size),
        "-ngl", str(args.n_gpu_layers),
        "-fa", args.flash_attn,
        "-ctk", variant["cache_k"],
        "-ctv", variant["cache_v"],
        "-f", str(prompt_file),
        "-n", str(n_predict),
        "--temp", "0",
        "--no-display-prompt",
        "--no-warmup",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.case_timeout,
        )
    finally:
        prompt_file.unlink(missing_ok=True)
    elapsed = time.time() - start
    return clean_completion(proc.stdout, prompt), proc.returncode, elapsed


def load_dataset_plan(mod, args, dataset_name: str) -> dict:
    dataset = mod.GpqaDataset(variant="diamond", seed=args.seed) if dataset_name == "gpqa" else mod.Gsm8kDataset()
    n_cases = args.gpqa_n_cases if dataset_name == "gpqa" else args.gsm8k_n_cases
    n_predict = args.gpqa_n_predict if dataset_name == "gpqa" else args.gsm8k_n_predict
    return {
        "dataset": dataset,
        "indices": pick_indices(len(dataset), n_cases, args.seed),
        "grader": mod.Grader(grader_type="regex", dataset_type=dataset_name),
        "n_predict": n_predict,
        "wilson_interval": mod.wilson_interval,
    }


def evaluate_dataset(args, variant_name: str, dataset_name: str, plan: dict, out_path: Path) -> None:
    dataset = plan["dataset"]
    indices = plan["indices"]
    grader = plan["grader"]
    n_predict = plan["n_predict"]
    variant = VARIANTS[variant_name]

    cases = {}
    correct = 0
    for count, idx in enumerate(indices, start=1):
        question = dataset.get_question(idx)
        prompt = dataset.get_prompt(question)
        expected = dataset.get_answer(question)
        response, returncode, elapsed = run_cli(args.llama_cli, variant, prompt, n_predict, args)
        trimmed = grader._truncate_response(response, max_lines=10)
        ok = returncode == 0
        is_correct, answer = grader.grade(expected, trimmed, prompt) if ok else (False, None)
        correct += int(is_correct)
        task_id = f"{dataset_name}_000_{idx:03d}"
        cases[task_id] = {
            "task_id": task_id,
            "prompt": prompt,
            "expected": expected,
            "response": response,
            "answer": answer,
            "grader_log": {"pred": trimmed, "grader_type": "regex", "returncode": returncode},
            "correct": is_correct,
            "status": "ok" if ok else f"error: returncode={returncode}",
            "tokens": n_predict,
            "tps_gen": n_predict / elapsed if elapsed > 0 else None,
            "t_gen_ms": elapsed * 1000.0,
            "server_name": variant_name,
            "chunk_idx": 0,
            "problem_idx": idx,
        }
        print(f"{variant_name} {dataset_name} {count}/{len(indices)} correct={correct} answer={answer} expected={expected}", flush=True)

    ci_lower, ci_upper = plan["wilson_interval"](correct, len(indices))
    out = {
        "id": dataset_name,
        "model_name": variant_name,
        "tasks": list(cases),
        "task_states": {
            "total": len(indices),
            "correct": correct,
            "total_time": sum(c.get("t_gen_ms", 0.0) for c in cases.values()) / 1000.0,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cases": cases,
        },
        "sampling_config": {"temperature": 0, "n_predict": n_predict, "ctx_size": args.ctx_size},
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/gpqa_gsm8k_cli_eval_current")
    parser.add_argument("--llama-cli", type=Path, default=ROOT / "third_party/OSCAR/build-cuda/bin/llama-completion")
    parser.add_argument("--variants", default="baseline_bf16,plain_int2,oscar_int2,oscar_int2_mixed,oscar_turbo2_streamk,turbo2_streamk,oscar_turbo3,plain_int3,oscar_int4,plain_int4")
    parser.add_argument("--datasets", default="gpqa,gsm8k")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
    parser.add_argument("--extra-env", default="", help="Comma-separated KEY=VALUE entries for the llama process.")
    parser.add_argument("--gpqa-n-cases", type=int, default=50)
    parser.add_argument("--gsm8k-n-cases", type=int, default=50)
    parser.add_argument("--gpqa-n-predict", type=int, default=96)
    parser.add_argument("--gsm8k-n-predict", type=int, default=128)
    parser.add_argument("--case-timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print planned work and exit; default on.")
    parser.add_argument("--real", action="store_false", dest="dry_run", help="Execute evaluation.")
    parser.add_argument("--ack-eval", action="store_true", help="Required with --real.")
    args = parser.parse_args()

    if not args.dry_run and not args.ack_eval:
        raise SystemExit("Refusing real eval without --ack-eval.")

    mod = load_eval_module()
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.txt").write_text(
        f"ctx_size={args.ctx_size}\n"
        f"gpqa_n_cases={args.gpqa_n_cases}\n"
        f"gsm8k_n_cases={args.gsm8k_n_cases}\n"
        f"seed={args.seed}\n"
        f"n_gpu_layers={args.n_gpu_layers}\n"
        f"flash_attn={args.flash_attn}\n"
        f"extra_env={args.extra_env}\n"
        "ctx_note=CLI accuracy eval uses 4096 context; 32k speed/memory is in runs/llamacpp_32k_kv_matrix_current.\n"
    )

    variants = [v for v in args.variants.split(",") if v]
    datasets = [d for d in args.datasets.split(",") if d]
    if args.dry_run:
        print("Dry run complete; pass --real --ack-eval to execute.")
        print(f"out_dir={args.out_dir}")
        print(f"variants={','.join(variants)}")
        print(f"datasets={','.join(datasets)}")
        print(f"llama_completion={args.llama_cli}")
        return

    dataset_plans = {dataset_name: load_dataset_plan(mod, args, dataset_name) for dataset_name in datasets}

    for variant_name in variants:
        if variant_name not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant_name}")
        for dataset_name in datasets:
            out_path = raw_dir / f"{variant_name}_{dataset_name}.json"
            if out_path.exists():
                print(f"Skipping existing {out_path}", flush=True)
                continue
            evaluate_dataset(args, variant_name, dataset_name, dataset_plans[dataset_name], out_path)


if __name__ == "__main__":
    main()
