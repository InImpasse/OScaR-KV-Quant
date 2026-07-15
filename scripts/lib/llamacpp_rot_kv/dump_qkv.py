from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.lib.llamacpp_rot_kv.errors import MissingDependencyError, ToolExecutionError
from scripts.lib.llamacpp_rot_kv.calibration import read_prompts
from scripts.lib.llamacpp_rot_kv.llama_paths import calibrator_runtime_env, resolve_calibrator_bin
from scripts.lib.llamacpp_rot_kv.oskv import load_oskv

META_RE = re.compile(r"^(?P<tensor>Qcur|Kcur|Vcur)-(?P<layer>\d+)\.(?P<idx>\d+)\.meta\.txt$")
OSKV_COMPLETE_RE = re.compile(
    r"oskv_dump_complete:\s+prompt=(\d+)\s+tokens=(\d+)\s+path=(.+)$"
)
DEFAULT_DUMP_WORKERS = 1
DEFAULT_KEEP_RAW_DUMPS = False
DEFAULT_MULTI_PROMPT_BATCH_SIZE = 8
DUMP_QKV_STORAGE_DTYPE = "bfloat16"


def qkv_numpy_to_storage_tensor(torch, arr: np.ndarray):
    """Persist Q/K/V dumps as bf16 for compact on-disk calibration dumps."""
    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return tensor.to(getattr(torch, DUMP_QKV_STORAGE_DTYPE))
_MULTI_PROMPT_HELP_CACHE: dict[str, bool] = {}
_OSKV_HELP_CACHE: dict[str, bool] = {}


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(f"torch is required to write .pt calibration dumps: {exc}") from exc
    return torch


def split_thread_budget(total_threads: int | None, workers: int) -> int:
    if total_threads is None:
        total_threads = os.cpu_count() or 1
    return max(1, int(total_threads) // max(1, workers))


def calibrator_supports_oskv_streaming(bin_path: str) -> bool:
    cached = _OSKV_HELP_CACHE.get(bin_path)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [bin_path, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=calibrator_runtime_env(bin_path),
        )
    except (OSError, subprocess.TimeoutExpired):
        _OSKV_HELP_CACHE[bin_path] = False
        return False

    help_text = f"{result.stdout}\n{result.stderr}".lower()
    supported = "--dump-format" in help_text and "oskv" in help_text
    _OSKV_HELP_CACHE[bin_path] = supported
    return supported


def calibrator_supports_multi_prompt(bin_path: str) -> bool:
    cached = _MULTI_PROMPT_HELP_CACHE.get(bin_path)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [bin_path, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=calibrator_runtime_env(bin_path),
        )
    except (OSError, subprocess.TimeoutExpired):
        _MULTI_PROMPT_HELP_CACHE[bin_path] = False
        return False

    help_text = f"{result.stdout}\n{result.stderr}".lower()
    supported = "--dataset" in help_text and "--dump-root" in help_text
    _MULTI_PROMPT_HELP_CACHE[bin_path] = supported
    return supported


def parse_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def load_f32_tensor(meta_path: Path) -> tuple[np.ndarray, list[int]]:
    meta = parse_meta(meta_path)
    if meta.get("type") != "f32":
        raise ValueError(f"{meta_path}: expected f32, got {meta.get('type')}")
    ne = [int(x) for x in meta["ne"].split(",")]
    bin_path = meta_path.with_suffix("").with_suffix(".bin")
    arr = np.fromfile(bin_path, dtype=np.float32)
    expected = int(np.prod(ne))
    if arr.size != expected:
        raise ValueError(f"{meta_path}: expected {expected} floats, got {arr.size}")
    return arr, ne


def tensor_to_tokens_heads_dim(meta_path: Path) -> np.ndarray:
    arr, ne = load_f32_tensor(meta_path)
    if ne[0] <= 0 or ne[1] <= 0 or ne[2] <= 0:
        raise ValueError(f"{meta_path}: unsupported tensor shape ne={ne}")
    return arr.reshape((ne[2], ne[1], ne[0])).copy()


def collect_prefill_pass(tensor_dir: Path) -> dict[int, dict[str, Path]]:
    grouped: dict[tuple[int, int], dict[str, Path]] = {}
    for meta_path in tensor_dir.glob("*.meta.txt"):
        match = META_RE.match(meta_path.name)
        if not match:
            continue
        tensor = match.group("tensor")
        layer = int(match.group("layer"))
        idx = int(match.group("idx"))
        grouped.setdefault((layer, idx), {})[tensor] = meta_path

    by_layer: dict[int, dict[str, Path]] = {}
    for (layer, _idx), paths in grouped.items():
        if not {"Qcur", "Kcur", "Vcur"}.issubset(paths):
            continue
        try:
            _, ne_q = load_f32_tensor(paths["Qcur"])
        except ValueError:
            continue
        n_tokens = ne_q[2]
        if n_tokens <= 1:
            continue
        current = by_layer.get(layer)
        if current is None:
            by_layer[layer] = paths
            continue
        _, ne_prev = load_f32_tensor(current["Qcur"])
        if n_tokens > ne_prev[2]:
            by_layer[layer] = paths
    return by_layer


@dataclass(slots=True)
class DumpQkvConfig:
    model: Path
    dataset: Path
    out_dir: Path
    options: dict
    max_prompts: int | None = None
    dump_token_budget: int | None = None
    calib_profile: str | None = None
    ctx: int = 4096
    predict: int = 1
    ngl: int = 999
    flash_attn: str = "on"
    cache_type_k: str = "bf16"
    cache_type_v: str = "bf16"
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    dump_workers: int = DEFAULT_DUMP_WORKERS
    use_multi_prompt: bool = True
    multi_prompt_batch_size: int = DEFAULT_MULTI_PROMPT_BATCH_SIZE
    resume_partial: bool = False
    keep_raw_dumps: bool = DEFAULT_KEEP_RAW_DUMPS
    overwrite: bool = False
    dry_run: bool = False


def append_optional_int_arg(cmd: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_calibrator_cmd(
    *,
    bin_path: str,
    model: Path,
    prompt: str | None,
    ctx: int,
    predict: int,
    ngl: int,
    flash_attn: str,
    cache_type_k: str,
    cache_type_v: str,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
    dataset: Path | None = None,
    dump_root: Path | None = None,
    max_prompts: int | None = None,
    dump_format: str | None = None,
    token_budget: int | None = None,
) -> list[str]:
    cmd = [
        bin_path,
        "-m",
        str(model),
        "-n",
        str(predict),
        "-c",
        str(ctx),
        "-ngl",
        str(ngl),
        "-fa",
        flash_attn,
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
        "--no-warmup",
        "--log-disable",
        "--verbosity",
        "0",
        "--tensor-filter",
        "Qcur",
        "--tensor-filter",
        "Kcur",
        "--tensor-filter",
        "Vcur",
    ]
    append_optional_int_arg(cmd, "-b", batch_size)
    append_optional_int_arg(cmd, "-ub", ubatch_size)
    append_optional_int_arg(cmd, "-t", threads)
    append_optional_int_arg(cmd, "-tb", threads_batch)

    if dataset is not None:
        cmd.extend(["--dataset", str(dataset)])
        if dump_root is not None:
            cmd.extend(["--dump-root", str(dump_root)])
        if max_prompts is not None:
            cmd.extend(["--max-prompts", str(max_prompts)])
        if token_budget is not None:
            cmd.extend(["--token-budget", str(token_budget)])
        if dump_format is not None:
            cmd.extend(["--dump-format", dump_format])
    else:
        cmd.extend(["-p", prompt or ""])
    return cmd


def _run_calibrator_streaming(
    cmd: list[str],
    env: dict[str, str],
    error_dump_dir: Path,
    *,
    line_handler: Callable[[str], None] | None = None,
) -> None:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  [calibrator] {line}", flush=True)
            if line_handler is not None:
                line_handler(line)
        stdout_lines.append(line)
    returncode = proc.wait()
    if returncode != 0:
        error_dump_dir.mkdir(parents=True, exist_ok=True)
        (error_dump_dir / "stdout.txt").write_text("\n".join(stdout_lines), encoding="utf-8", errors="replace")
        (error_dump_dir / "stderr.txt").write_text("\n".join(stderr_lines), encoding="utf-8", errors="replace")
        raise ToolExecutionError(
            f"llama-rot-kv-calibrate failed with exit code {returncode}; see {error_dump_dir}"
        )


def run_calibrator(
    *,
    bin_path: str,
    model: Path,
    prompt: str,
    dump_dir: Path,
    ctx: int,
    predict: int,
    ngl: int,
    flash_attn: str,
    cache_type_k: str,
    cache_type_v: str,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
) -> None:
    env = calibrator_runtime_env(bin_path)
    env["LLAMA_DEBUG_TENSOR_DUMP_DIR"] = str(dump_dir)
    env["LLAMA_DEBUG_TENSOR_DUMP_ONLY"] = "1"
    cmd = _build_calibrator_cmd(
        bin_path=bin_path,
        model=model,
        prompt=prompt,
        ctx=ctx,
        predict=predict,
        ngl=ngl,
        flash_attn=flash_attn,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch,
    )
    _run_calibrator_streaming(cmd, env, dump_dir)


def run_calibrator_multi(
    *,
    bin_path: str,
    model: Path,
    dataset: Path,
    dump_root: Path,
    ctx: int,
    predict: int,
    ngl: int,
    flash_attn: str,
    cache_type_k: str,
    cache_type_v: str,
    max_prompts: int | None,
    token_budget: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
) -> None:
    dump_root.mkdir(parents=True, exist_ok=True)
    env = calibrator_runtime_env(bin_path)
    env["LLAMA_DEBUG_TENSOR_DUMP_ONLY"] = "1"
    cmd = _build_calibrator_cmd(
        bin_path=bin_path,
        model=model,
        prompt=None,
        ctx=ctx,
        predict=predict,
        ngl=ngl,
        flash_attn=flash_attn,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch,
        dataset=dataset,
        dump_root=dump_root,
        max_prompts=max_prompts,
        token_budget=token_budget,
    )
    _run_calibrator_streaming(cmd, env, dump_root)


def _cleanup_raw_prompt_dir(prompt_raw: Path) -> None:
    if prompt_raw.is_dir():
        shutil.rmtree(prompt_raw, ignore_errors=True)


def prompt_batches(prompts: list[str], batch_size: int) -> list[tuple[int, list[str]]]:
    if batch_size < 1:
        return [(0, prompts)]
    return [(first, prompts[first : first + batch_size]) for first in range(0, len(prompts), batch_size)]


def _write_batch_dataset(path: Path, prompts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n" for prompt in prompts),
        encoding="utf-8",
    )


def completed_prompt_count(out_dump: Path) -> int:
    layer_dirs = sorted(
        [path for path in out_dump.glob("layer_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_", 1)[1]),
    )
    if not layer_dirs:
        return 0

    count = 0
    while True:
        chunk_id = count + 1
        if not all(
            (layer_dir / tensor_name / f"{chunk_id}.pt").is_file()
            for layer_dir in layer_dirs
            for tensor_name in ("q", "k", "v", "seq_lens")
        ):
            break
        count += 1
    return count


def restore_prompt_log(out_dump: Path, prompts: list[str], count: int) -> None:
    prompts_path = out_dump / "prompts.jsonl"
    prompts_path.write_text(
        "".join(
            json.dumps({"chunk": index, "prompt": prompt}, ensure_ascii=False) + "\n"
            for index, prompt in enumerate(prompts[:count], start=1)
        ),
        encoding="utf-8",
    )


def write_chunk_from_arrays(
    torch,
    prompt_idx: int,
    prompt: str,
    layer_arrays: dict[int, dict[str, np.ndarray]],
    out_dump: Path,
) -> int:
    token_count: int | None = None
    for layer, arrays in sorted(layer_arrays.items()):
        q = arrays["Qcur"]
        k = arrays["Kcur"]
        v = arrays["Vcur"]
        for name, arr in (("q", q), ("k", k), ("v", v)):
            n_bad = int((~np.isfinite(arr)).sum())
            if n_bad:
                raise ValueError(
                    f"prompt {prompt_idx} layer {layer} {name}: dumped tensor contains {n_bad} non-finite values"
                )
        if token_count is None:
            token_count = int(q.shape[0])
        if int(q.shape[0]) != token_count or int(k.shape[0]) != token_count or int(v.shape[0]) != token_count:
            raise ValueError(f"layer {layer}: inconsistent token counts for prompt {prompt_idx}")

        layer_dir = out_dump / f"layer_{layer}"
        for name, arr in (("q", q), ("k", k), ("v", v)):
            sub = layer_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            torch.save(qkv_numpy_to_storage_tensor(torch, arr), sub / f"{prompt_idx}.pt")
        seq_dir = layer_dir / "seq_lens"
        seq_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor([token_count], dtype=torch.int32), seq_dir / f"{prompt_idx}.pt")

    if token_count is None:
        raise ValueError(f"prompt {prompt_idx}: no complete Q/K/V layer dumps found")
    with (out_dump / "prompts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"chunk": prompt_idx, "tokens": token_count, "prompt": prompt}, ensure_ascii=False) + "\n"
        )
    return token_count


def write_chunk(
    torch,
    prompt_idx: int,
    prompt: str,
    layer_paths: dict[int, dict[str, Path]],
    out_dump: Path,
    *,
    keep_raw_dumps: bool,
    prompt_raw: Path | None = None,
) -> int:
    token_count: int | None = None
    for layer, paths in sorted(layer_paths.items()):
        q = tensor_to_tokens_heads_dim(paths["Qcur"])
        k = tensor_to_tokens_heads_dim(paths["Kcur"])
        v = tensor_to_tokens_heads_dim(paths["Vcur"])
        for name, arr in (("q", q), ("k", k), ("v", v)):
            n_bad = int((~np.isfinite(arr)).sum())
            if n_bad:
                raise ValueError(
                    f"prompt {prompt_idx} layer {layer} {name}: dumped tensor contains {n_bad} non-finite values"
                )
        if token_count is None:
            token_count = int(q.shape[0])
        if int(q.shape[0]) != token_count or int(k.shape[0]) != token_count or int(v.shape[0]) != token_count:
            raise ValueError(f"layer {layer}: inconsistent token counts for prompt {prompt_idx}")

        layer_dir = out_dump / f"layer_{layer}"
        for name, arr in (("q", q), ("k", k), ("v", v)):
            sub = layer_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            torch.save(qkv_numpy_to_storage_tensor(torch, arr), sub / f"{prompt_idx}.pt")
        seq_dir = layer_dir / "seq_lens"
        seq_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor([token_count], dtype=torch.int32), seq_dir / f"{prompt_idx}.pt")

    if token_count is None:
        raise ValueError(f"prompt {prompt_idx}: no complete Q/K/V layer dumps found")
    with (out_dump / "prompts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"chunk": prompt_idx, "tokens": token_count, "prompt": prompt}, ensure_ascii=False) + "\n"
        )

    if not keep_raw_dumps and prompt_raw is not None:
        _cleanup_raw_prompt_dir(prompt_raw)
    return token_count


def _convert_oskv_prompt(
    torch,
    *,
    oskv_path: Path,
    prompt_idx: int,
    prompt: str,
    out_dump: Path,
    keep_raw_dumps: bool,
) -> int:
    layer_arrays = load_oskv(oskv_path)
    tokens = write_chunk_from_arrays(torch, prompt_idx, prompt, layer_arrays, out_dump)
    if not keep_raw_dumps:
        oskv_path.unlink(missing_ok=True)
    return tokens


def run_calibrator_oskv_streaming(
    *,
    bin_path: str,
    model: Path,
    dataset: Path,
    dump_root: Path,
    ctx: int,
    predict: int,
    ngl: int,
    flash_attn: str,
    cache_type_k: str,
    cache_type_v: str,
    max_prompts: int | None,
    token_budget: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
    line_handler: Callable[[str], None] | None = None,
) -> None:
    dump_root.mkdir(parents=True, exist_ok=True)
    env = calibrator_runtime_env(bin_path)
    cmd = _build_calibrator_cmd(
        bin_path=bin_path,
        model=model,
        prompt=None,
        ctx=ctx,
        predict=predict,
        ngl=ngl,
        flash_attn=flash_attn,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        threads_batch=threads_batch or threads,
        dataset=dataset,
        dump_root=dump_root,
        max_prompts=max_prompts,
        dump_format="oskv",
        token_budget=token_budget,
    )
    _run_calibrator_streaming(cmd, env, dump_root, line_handler=line_handler)


def _dump_oskv_streaming(
    *,
    torch,
    calibrator: str,
    config: DumpQkvConfig,
    prompts: list[str],
    out_dump: Path,
    raw_dir: Path,
    threads_per_worker: int,
    prompt_offset: int = 0,
    total_prompts: int | None = None,
) -> tuple[int, int, float, float]:
    dataset_path = config.out_dir / ".oskv_dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    dumped_tokens = 0
    captured_prompts = 0
    convert_s = 0.0
    all_prompts = total_prompts or (prompt_offset + len(prompts))

    def on_line(line: str) -> None:
        nonlocal dumped_tokens, captured_prompts, convert_s
        match = OSKV_COMPLETE_RE.search(line)
        if not match:
            return
        prompt_idx = int(match.group(1))
        oskv_path = Path(match.group(3).strip())
        if not (1 <= prompt_idx <= len(prompts)):
            raise ToolExecutionError(f"unexpected prompt index from calibrator: {prompt_idx}")
        prompt = prompts[prompt_idx - 1]
        prompt_idx += prompt_offset
        convert_started = time.perf_counter()
        tokens = _convert_oskv_prompt(
            torch,
            oskv_path=oskv_path,
            prompt_idx=prompt_idx,
            prompt=prompt,
            out_dump=out_dump,
            keep_raw_dumps=config.keep_raw_dumps,
        )
        convert_s += time.perf_counter() - convert_started
        dumped_tokens += tokens
        captured_prompts += 1
        print(f"dumped prompt {prompt_idx}/{all_prompts}: tokens={tokens}", flush=True)

    capture_started = time.perf_counter()
    run_calibrator_oskv_streaming(
        bin_path=calibrator,
        model=config.model,
        dataset=dataset_path,
        dump_root=raw_dir,
        ctx=config.ctx,
        predict=config.predict,
        ngl=config.ngl,
        flash_attn=config.flash_attn,
        cache_type_k=config.cache_type_k,
        cache_type_v=config.cache_type_v,
        max_prompts=len(prompts),
        token_budget=config.dump_token_budget,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        threads=threads_per_worker,
        threads_batch=config.threads_batch or threads_per_worker,
        line_handler=on_line,
    )
    capture_s = time.perf_counter() - capture_started
    dataset_path.unlink(missing_ok=True)
    if not config.keep_raw_dumps:
        shutil.rmtree(raw_dir, ignore_errors=True)
    if captured_prompts != len(prompts):
        if config.dump_token_budget is None:
            raise ToolExecutionError(
                f"expected {len(prompts)} OSKV prompts, converted {captured_prompts}"
            )
        if captured_prompts == 0:
            raise ToolExecutionError("token-budget dump produced zero prompts")
        if dumped_tokens < int(config.dump_token_budget * 0.9):
            raise ToolExecutionError(
                f"token-budget dump captured only {dumped_tokens} tokens "
                f"(budget={config.dump_token_budget})"
            )
    return dumped_tokens, captured_prompts, capture_s, convert_s


def _capture_prompt(
    *,
    calibrator: str,
    config: DumpQkvConfig,
    prompt_idx: int,
    prompt: str,
    prompt_raw: Path,
    threads_per_worker: int,
) -> tuple[int, str, Path]:
    prompt_raw.mkdir(parents=True, exist_ok=True)
    run_calibrator(
        bin_path=calibrator,
        model=config.model,
        prompt=prompt,
        dump_dir=prompt_raw,
        ctx=config.ctx,
        predict=config.predict,
        ngl=config.ngl,
        flash_attn=config.flash_attn,
        cache_type_k=config.cache_type_k,
        cache_type_v=config.cache_type_v,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        threads=threads_per_worker,
        threads_batch=config.threads_batch or threads_per_worker,
    )
    return prompt_idx, prompt, prompt_raw


def _capture_prompts(
    *,
    calibrator: str,
    config: DumpQkvConfig,
    prompts: list[str],
    raw_dir: Path,
    threads_per_worker: int,
    dump_workers: int,
) -> list[tuple[int, str, Path]]:
    items = [(index, prompt, raw_dir / f"prompt_{index:05d}") for index, prompt in enumerate(prompts, start=1)]

    if dump_workers <= 1:
        return [
            _capture_prompt(
                calibrator=calibrator,
                config=config,
                prompt_idx=prompt_idx,
                prompt=prompt,
                prompt_raw=prompt_raw,
                threads_per_worker=threads_per_worker,
            )
            for prompt_idx, prompt, prompt_raw in items
        ]

    captured: list[tuple[int, str, Path]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=dump_workers) as executor:
        for prompt_idx, prompt, prompt_raw in items:
            future = executor.submit(
                _capture_prompt,
                calibrator=calibrator,
                config=config,
                prompt_idx=prompt_idx,
                prompt=prompt,
                prompt_raw=prompt_raw,
                threads_per_worker=threads_per_worker,
            )
            futures[future] = prompt_idx
        for future in as_completed(futures):
            captured.append(future.result())
    captured.sort(key=lambda item: item[0])
    return captured


def _dump_multi_prompt_batches(
    *,
    torch,
    calibrator: str,
    config: DumpQkvConfig,
    prompts: list[str],
    out_dump: Path,
    raw_dir: Path,
    threads_per_worker: int,
    prompt_offset: int = 0,
    total_prompts: int | None = None,
) -> tuple[int, int, float, float]:
    batch_size = max(1, int(config.multi_prompt_batch_size))
    batch_dataset = config.out_dir / ".multi_prompt_batch.jsonl"
    dumped_tokens = 0
    captured_prompts = 0
    capture_s = 0.0
    convert_s = 0.0

    for batch_number, (first, batch_prompts) in enumerate(prompt_batches(prompts, batch_size), start=1):
        batch_raw_dir = raw_dir / f"batch_{batch_number:04d}"
        _write_batch_dataset(batch_dataset, batch_prompts)
        first_prompt_index = prompt_offset + first + 1
        all_prompts = total_prompts or len(prompts)
        print(
            f"capturing batch {batch_number}: prompts "
            f"{first_prompt_index}-{first_prompt_index + len(batch_prompts) - 1}/{all_prompts}",
            flush=True,
        )
        capture_started = time.perf_counter()
        run_calibrator_multi(
            bin_path=calibrator,
            model=config.model,
            dataset=batch_dataset,
            dump_root=batch_raw_dir,
            ctx=config.ctx,
            predict=config.predict,
            ngl=config.ngl,
            flash_attn=config.flash_attn,
            cache_type_k=config.cache_type_k,
            cache_type_v=config.cache_type_v,
            max_prompts=len(batch_prompts),
            batch_size=config.batch_size,
            ubatch_size=config.ubatch_size,
            threads=threads_per_worker,
            threads_batch=config.threads_batch or threads_per_worker,
        )
        capture_s += time.perf_counter() - capture_started

        convert_started = time.perf_counter()
        for batch_index, prompt in enumerate(batch_prompts, start=1):
            prompt_index = prompt_offset + first + batch_index
            prompt_raw = batch_raw_dir / f"prompt_{batch_index:05d}"
            tokens = write_chunk(
                torch,
                prompt_index,
                prompt,
                collect_prefill_pass(prompt_raw),
                out_dump,
                keep_raw_dumps=config.keep_raw_dumps,
                prompt_raw=prompt_raw,
            )
            dumped_tokens += tokens
            captured_prompts += 1
            print(f"dumped prompt {prompt_index}/{all_prompts}: tokens={tokens}", flush=True)
        convert_s += time.perf_counter() - convert_started

        if not config.keep_raw_dumps:
            shutil.rmtree(batch_raw_dir, ignore_errors=True)

    batch_dataset.unlink(missing_ok=True)
    if not config.keep_raw_dumps:
        shutil.rmtree(raw_dir, ignore_errors=True)
    return dumped_tokens, captured_prompts, capture_s, convert_s


def dump_qkv(config: DumpQkvConfig) -> Path:
    prompts = read_prompts(config.dataset, config.max_prompts)
    out_dir = config.out_dir.resolve()
    out_dump = out_dir / "qkv_dumps" / "llamacpp"
    raw_dir = out_dir / "raw_llama_debug"
    calibrator = resolve_calibrator_bin(config.options)
    dump_workers = max(1, int(config.dump_workers))
    threads_per_worker = split_thread_budget(config.threads, dump_workers)
    oskv_streaming = (
        config.use_multi_prompt
        and dump_workers == 1
        and calibrator_supports_oskv_streaming(calibrator)
    )
    multi_prompt = (
        config.use_multi_prompt
        and dump_workers == 1
        and not oskv_streaming
        and calibrator_supports_multi_prompt(calibrator)
    )
    if oskv_streaming:
        dump_mode = "oskv_streaming"
    elif multi_prompt:
        dump_mode = f"multi_prompt_batches={max(1, int(config.multi_prompt_batch_size))}"
    else:
        dump_mode = f"workers={dump_workers}"

    if config.dry_run:
        print(
            f"would dump {len(prompts)} prompts from {config.dataset} "
            f"mode={dump_mode} threads_per_worker={threads_per_worker} "
            f"keep_raw_dumps={config.keep_raw_dumps}"
        )
        print(f"model={config.model}")
        print(f"llama-rot-kv-calibrate={calibrator}")
        print(f"out={out_dump}")
        return out_dump

    if not Path(calibrator).is_file():
        raise MissingDependencyError(f"missing llama-rot-kv-calibrate binary: {calibrator}")
    if not config.model.is_file():
        raise FileNotFoundError(f"missing model GGUF: {config.model}")
    if out_dir.exists() and any(out_dir.iterdir()) and not config.overwrite and not config.resume_partial:
        raise FileExistsError(
            f"output dir is not empty, pass overwrite=true to replace: {out_dir}"
        )

    torch = import_torch()
    if config.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dump.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prompt_offset = completed_prompt_count(out_dump) if config.resume_partial else 0
    if prompt_offset > len(prompts):
        raise ValueError(
            f"existing dump contains {prompt_offset} prompts, but dataset contains only {len(prompts)} prompts"
        )
    remaining_prompts = prompts[prompt_offset:]
    if prompt_offset:
        restore_prompt_log(out_dump, prompts, prompt_offset)
        print(f"resuming partial dump from prompt {prompt_offset + 1}/{len(prompts)}", flush=True)

    batch_size = config.batch_size or config.ctx
    ubatch_size = config.ubatch_size or min(batch_size, 512)
    config.batch_size = batch_size
    config.ubatch_size = ubatch_size

    capture_started = time.perf_counter()
    if oskv_streaming:
        dumped_tokens, ok_prompts, capture_s, convert_s = _dump_oskv_streaming(
            torch=torch,
            calibrator=calibrator,
            config=config,
            prompts=remaining_prompts,
            out_dump=out_dump,
            raw_dir=raw_dir,
            threads_per_worker=threads_per_worker,
            prompt_offset=prompt_offset,
            total_prompts=len(prompts),
        )
    elif multi_prompt:
        dumped_tokens, ok_prompts, capture_s, convert_s = _dump_multi_prompt_batches(
            torch=torch,
            calibrator=calibrator,
            config=config,
            prompts=remaining_prompts,
            out_dump=out_dump,
            raw_dir=raw_dir,
            threads_per_worker=threads_per_worker,
            prompt_offset=prompt_offset,
            total_prompts=len(prompts),
        )
    else:
        captured = _capture_prompts(
            calibrator=calibrator,
            config=config,
            prompts=remaining_prompts,
            raw_dir=raw_dir,
            threads_per_worker=threads_per_worker,
            dump_workers=dump_workers,
        )
        capture_s = time.perf_counter() - capture_started

        convert_started = time.perf_counter()
        dumped_tokens = 0
        ok_prompts = 0
        for prompt_idx, prompt, prompt_raw in captured:
            prompt_idx += prompt_offset
            layer_paths = collect_prefill_pass(prompt_raw)
            tokens = write_chunk(
                torch,
                prompt_idx,
                prompt,
                layer_paths,
                out_dump,
                keep_raw_dumps=config.keep_raw_dumps,
                prompt_raw=prompt_raw,
            )
            dumped_tokens += tokens
            ok_prompts += 1
            print(f"dumped prompt {prompt_idx}/{len(prompts)}: tokens={tokens} layers={len(layer_paths)}")
        convert_s = time.perf_counter() - convert_started

    meta = {
        "format_version": 1,
        "source": "llama.cpp llama-rot-kv-calibrate tensor dump",
        "model": str(config.model),
        "dataset": str(config.dataset),
        "num_prompts_requested": len(prompts),
        "num_prompts_captured": prompt_offset + ok_prompts,
        "dumped_tokens": dumped_tokens,
        "dump_mode": dump_mode,
        "dump_backend": "oskv" if oskv_streaming else "legacy_raw",
        "dump_token_budget": config.dump_token_budget,
        "calib_profile": config.calib_profile,
        "dump_workers": dump_workers,
        "multi_prompt_batch_size": config.multi_prompt_batch_size if multi_prompt else None,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "resumed_prompts": prompt_offset,
        "threads_per_worker": threads_per_worker,
        "keep_raw_dumps": config.keep_raw_dumps,
        "dump_dtype": DUMP_QKV_STORAGE_DTYPE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "calibration_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out_dir / 'calibration_meta.json'} "
        f"mode={dump_mode} capture={capture_s:.1f}s convert={convert_s:.1f}s"
    )
    return out_dump
