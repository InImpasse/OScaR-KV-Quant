#!/usr/bin/env python3
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS = [
    Path("README.md"),
    Path("AGENTS.md"),
    Path("docs"),
    Path("scripts"),
    Path("third_party/OSCAR"),
    Path("runs/llamacpp_32k_kv_matrix_current"),
    Path("runs/cuda_graph_ab_512_current"),
]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "build",
    "build-cuda",
    "CMakeFiles",
    "vendor",
    "raw",
}

UPSTREAM_LLAMA_CPP_NOISE_DIRS = {
    "common",
    "conversion",
    "docs",
    "examples",
    "grammars",
    "models",
    "scripts",
    "src",
    "tests",
    "tools",
}

SKIP_FILE_NAMES = {
    "check_llamacpp_only.py",
}

FORBIDDEN = [
    "s" + "glang",
    "S" + "GLang",
    "S" + "GLANG",
    "InImpasse",
    "server_args",
    "RuntimeEndpoint",
    "server" + " harness",
    "server" + "-harness",
]

REQUIRED_LLAMA_CPP_SIGNALS = [
    (Path("README.md"), "llama.cpp"),
    (Path("AGENTS.md"), "llama.cpp"),
    (Path("docs/LLAMACPP_32K_KV_TEST_PLAN.md"), "llama.cpp"),
    (Path("third_party/OSCAR/README.md"), "llama.cpp"),
]


def iter_files(target: Path):
    path = ROOT / target
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.name not in SKIP_FILE_NAMES:
            yield path
        return
    for child in path.rglob("*"):
        if child.is_dir():
            continue
        rel_parts = child.relative_to(ROOT).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if len(rel_parts) >= 3 and rel_parts[0] == "third_party" and rel_parts[1] == "OSCAR":
            if rel_parts[2] in UPSTREAM_LLAMA_CPP_NOISE_DIRS:
                continue
        if child.name in SKIP_FILE_NAMES:
            continue
        yield child


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except UnicodeDecodeError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that this workspace stays llama.cpp-only.")
    parser.add_argument("targets", nargs="*", type=Path, default=DEFAULT_TARGETS)
    args = parser.parse_args()

    hits = []
    for target in args.targets:
        for path in iter_files(target):
            text = read_text(path)
            for needle in FORBIDDEN:
                if needle in text:
                    rel = path.relative_to(ROOT)
                    hits.append(f"{rel}: contains forbidden non-llama.cpp marker: {needle}")

    missing = []
    for rel_path, needle in REQUIRED_LLAMA_CPP_SIGNALS:
        path = ROOT / rel_path
        if not path.exists() or needle not in read_text(path):
            missing.append(f"{rel_path}: missing required marker {needle!r}")

    if hits or missing:
        for line in hits + missing:
            print(line)
        raise SystemExit(1)

    print("llama.cpp-only static checks passed")


if __name__ == "__main__":
    main()
