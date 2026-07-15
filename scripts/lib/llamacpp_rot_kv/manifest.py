from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
MANIFEST_DIRNAME = "stage_manifests"


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dataset_fingerprint(path: Path, max_prompts: int | None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "max_prompts": max_prompts,
    }


def manifest_dir(work_dir: Path) -> Path:
    return work_dir / MANIFEST_DIRNAME


def manifest_path(work_dir: Path, stage: str) -> Path:
    return manifest_dir(work_dir) / f"{stage}.json"


def write_manifest(work_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    path = manifest_path(work_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format_version": MANIFEST_VERSION,
        "stage": stage,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(work_dir: Path, stage: str) -> dict[str, Any] | None:
    path = manifest_path(work_dir, stage)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def manifests_match(stored: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if stored is None:
        return False
    ignore = {"written_at", "format_version", "stage"}
    comparable = {key: value for key, value in stored.items() if key not in ignore}
    return comparable == expected


def count_layer_dirs(dump_path: Path) -> int:
    if not dump_path.is_dir():
        return 0
    return sum(1 for entry in dump_path.iterdir() if entry.is_dir() and entry.name.startswith("layer_"))


def count_prompt_chunks(dump_path: Path) -> int:
    prompts_path = dump_path / "prompts.jsonl"
    if not prompts_path.is_file():
        return 0
    return sum(1 for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip())


def dump_stage_artifacts_valid(
    work_dir: Path,
    *,
    expected_prompts: int,
    require_calibration_meta: bool = True,
) -> bool:
    dump_root = work_dir / "qkv_dumps" / "llamacpp"
    if not dump_root.is_dir():
        return False

    if count_prompt_chunks(dump_root) != expected_prompts:
        return False
    if count_layer_dirs(dump_root) == 0:
        return False

    for layer_dir in dump_root.iterdir():
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        q_dir = layer_dir / "q"
        if not q_dir.is_dir():
            return False
        if len(list(q_dir.glob("*.pt"))) < expected_prompts:
            return False

    if require_calibration_meta:
        meta_path = work_dir / "calibration_meta.json"
        if not meta_path.is_file():
            return False
    return True


def rotation_stage_artifacts_valid(rot_dir: Path) -> bool:
    if not rot_dir.is_dir():
        return False
    k_path = rot_dir / "k_rotation_qqt_r_h_pbr.pt"
    v_path = rot_dir / "v_rotation_sst_r_h_pbr.pt"
    meta_path = rot_dir / "rotation_meta.json"
    return k_path.is_file() and v_path.is_file() and meta_path.is_file()
