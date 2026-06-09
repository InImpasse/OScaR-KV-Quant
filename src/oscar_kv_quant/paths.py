"""Repo-relative path helpers."""

from __future__ import annotations

from pathlib import Path

from oscar_kv_quant.profiles import repo_root


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def to_repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    root = repo_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def repo_cache_dir() -> Path:
    cache = repo_root() / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache
