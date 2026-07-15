from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.lib.llamacpp_rot_kv.bake_gguf import BakeGgufConfig, bake_rotations_to_gguf, derive_output_path
from scripts.lib.llamacpp_rot_kv.calibration import read_prompts
from scripts.lib.llamacpp_rot_kv.compute_rotation import ComputeRotationConfig, compute_rotations
from scripts.lib.llamacpp_rot_kv.dump_qkv import DumpQkvConfig, dump_qkv
from scripts.lib.llamacpp_rot_kv.llama_paths import DEFAULT_DATASET
from scripts.lib.llamacpp_rot_kv.manifest import (
    dataset_fingerprint,
    dump_stage_artifacts_valid,
    file_fingerprint,
    load_manifest,
    manifests_match,
    rotation_stage_artifacts_valid,
    write_manifest,
)


@dataclass(slots=True)
class PipelineConfig:
    base_gguf: Path
    dataset: Path = DEFAULT_DATASET
    work_dir: Path = Path("runs/llamacpp_rot_kv_calibration")
    output_gguf: Path | None = None
    max_prompts: int | None = None
    dump_token_budget: int | None = None
    calib_profile: str | None = None
    ctx: int = 4096
    head_dim: int = 128
    composition: str = "r_h_pbr"
    method: str = "qqt_sst"
    chunk_id: str | int = "all"
    predict: int = 1
    ngl: int = 999
    flash_attn: str = "on"
    cache_type_k: str = "bf16"
    cache_type_v: str = "bf16"
    dump_workers: int = 1
    threads: int | None = None
    multi_prompt_batch_size: int = 8
    keep_raw_dumps: bool = False
    allow_layer_mismatch: bool = False
    replace_rotations: bool = True
    max_orthogonality_error: float = 1e-4
    skip_dump: bool = False
    skip_compute: bool = False
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False
    calibrator_options: dict = field(default_factory=dict)


def _write_timings(work_dir: Path, timings: dict[str, float | str | int | bool | None]) -> None:
    path = work_dir / "timings.json"
    path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")


def _build_dump_manifest(
    *,
    base: Path,
    dataset: Path,
    max_prompts: int | None,
    ctx: int,
    dump_workers: int,
    threads: int | None,
    multi_prompt_batch_size: int,
    keep_raw_dumps: bool,
    head_dim: int,
    dump_token_budget: int | None,
) -> dict:
    return {
        "base_gguf": file_fingerprint(base),
        "dataset": dataset_fingerprint(dataset, max_prompts),
        "ctx": ctx,
        "dump_workers": dump_workers,
        "threads": threads,
        "multi_prompt_batch_size": multi_prompt_batch_size,
        "keep_raw_dumps": keep_raw_dumps,
        "head_dim": head_dim,
        "dump_token_budget": dump_token_budget,
    }


def _build_rotation_manifest(
    *,
    dump_manifest: dict,
    composition: str,
    method: str,
    head_dim: int,
    chunk_id: str | int,
) -> dict:
    return {
        "dump": dump_manifest,
        "composition": composition,
        "method": method,
        "head_dim": head_dim,
        "chunk_id": chunk_id,
    }


def run_llamacpp_rot_kv_pipeline(config: PipelineConfig) -> Path:
    work_dir = config.work_dir.resolve()
    dump_path = work_dir / "qkv_dumps" / "llamacpp"
    rot_dir = work_dir / "rotations"
    calibration_meta = work_dir / "calibration_meta.json"
    base = config.base_gguf.resolve()
    dataset = config.dataset.resolve()
    output_path = (config.output_gguf or derive_output_path(base)).resolve()

    expected_prompts = len(read_prompts(dataset, max_prompts=config.max_prompts))
    dump_manifest = _build_dump_manifest(
        base=base,
        dataset=dataset,
        max_prompts=config.max_prompts,
        ctx=config.ctx,
        dump_workers=config.dump_workers,
        threads=config.threads,
        multi_prompt_batch_size=config.multi_prompt_batch_size,
        keep_raw_dumps=config.keep_raw_dumps,
        head_dim=config.head_dim,
        dump_token_budget=config.dump_token_budget,
    )
    rotation_manifest = _build_rotation_manifest(
        dump_manifest=dump_manifest,
        composition=config.composition,
        method=config.method,
        head_dim=config.head_dim,
        chunk_id=config.chunk_id,
    )
    timings: dict[str, float | str | int | bool | None] = {
        "dry_run": config.dry_run,
        "dump_workers": config.dump_workers,
        "threads": config.threads,
        "multi_prompt_batch_size": config.multi_prompt_batch_size,
        "keep_raw_dumps": config.keep_raw_dumps,
        "dump_token_budget": config.dump_token_budget,
    }

    can_resume_dump = (
        config.resume
        and not config.skip_dump
        and manifests_match(load_manifest(work_dir, "dump"), dump_manifest)
        and dump_stage_artifacts_valid(work_dir, expected_prompts=expected_prompts)
    )
    if config.skip_dump or can_resume_dump:
        if not dump_path.is_dir() and not config.dry_run:
            raise FileNotFoundError(f"dump stage skipped but missing dump path: {dump_path}")
        timings["dump_s"] = 0.0
        timings["dump_resumed"] = can_resume_dump
        print(f"Skipping dump stage (resume={can_resume_dump}, skip_dump={config.skip_dump})")
    else:
        dump_started = time.perf_counter()
        dump_qkv(
            DumpQkvConfig(
                model=base,
                dataset=dataset,
                out_dir=work_dir,
                options=dict(config.calibrator_options),
                max_prompts=config.max_prompts,
                dump_token_budget=config.dump_token_budget,
                calib_profile=config.calib_profile,
                ctx=config.ctx,
                predict=config.predict,
                ngl=config.ngl,
                flash_attn=config.flash_attn,
                cache_type_k=config.cache_type_k,
                cache_type_v=config.cache_type_v,
                threads=config.threads,
                dump_workers=config.dump_workers,
                multi_prompt_batch_size=config.multi_prompt_batch_size,
                keep_raw_dumps=config.keep_raw_dumps,
                overwrite=config.overwrite,
                dry_run=config.dry_run,
            )
        )
        timings["dump_s"] = round(time.perf_counter() - dump_started, 3)
        if not config.dry_run:
            write_manifest(work_dir, "dump", dump_manifest)

    can_resume_rotation = (
        config.resume
        and not config.skip_compute
        and manifests_match(load_manifest(work_dir, "rotation"), rotation_manifest)
        and rotation_stage_artifacts_valid(rot_dir)
    )
    if config.skip_compute or can_resume_rotation:
        if not rot_dir.is_dir() and not config.dry_run:
            raise FileNotFoundError(f"rotation stage skipped but missing rotation dir: {rot_dir}")
        timings["rotation_s"] = 0.0
        timings["rotation_resumed"] = can_resume_rotation
        print(f"Skipping rotation stage (resume={can_resume_rotation}, skip_compute={config.skip_compute})")
    else:
        rotation_started = time.perf_counter()
        compute_rotations(
            ComputeRotationConfig(
                dump_path=dump_path,
                output_dir=rot_dir,
                head_dim=config.head_dim,
                chunk_id=config.chunk_id,
                method=config.method,
                composition=config.composition,
                calibration_meta=calibration_meta if calibration_meta.is_file() else None,
                dry_run=config.dry_run,
            )
        )
        timings["rotation_s"] = round(time.perf_counter() - rotation_started, 3)
        if not config.dry_run:
            write_manifest(work_dir, "rotation", rotation_manifest)

    bake_started = time.perf_counter()
    baked_path = bake_rotations_to_gguf(
        BakeGgufConfig(
            base=base,
            rot_dir=rot_dir,
            out_path=output_path,
            k_rotation_filename=f"k_rotation_qqt_{config.composition}.pt",
            v_rotation_filename=f"v_rotation_sst_{config.composition}.pt",
            max_orthogonality_error=config.max_orthogonality_error,
            allow_layer_mismatch=config.allow_layer_mismatch,
            replace_rotations=config.replace_rotations,
            overwrite=config.overwrite,
            dry_run=config.dry_run,
        )
    )
    timings["bake_s"] = round(time.perf_counter() - bake_started, 3)
    timings["total_s"] = round(
        float(timings.get("dump_s", 0.0))
        + float(timings.get("rotation_s", 0.0))
        + float(timings.get("bake_s", 0.0)),
        3,
    )
    _write_timings(work_dir, timings)
    return baked_path
