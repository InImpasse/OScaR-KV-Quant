from __future__ import annotations

import os
import platform
from pathlib import Path

from scripts.lib.llamacpp_rot_kv.errors import MissingDependencyError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OSCAR_ROOT = REPO_ROOT / "third_party" / "OSCAR"
GGUF_PY_ROOT = DEFAULT_OSCAR_ROOT / "gguf-py"
DEFAULT_DATASET = REPO_ROOT / "data" / "calibration_prompts_gpqa_gsm8k.jsonl"
CALIBRATOR_NAME = "llama-rot-kv-calibrate"


def _platform_name(system: str | None = None) -> str:
    name = (system or platform.system()).lower()
    if name.startswith("win"):
        return "windows"
    if name.startswith("linux"):
        return "linux"
    if name.startswith("darwin") or name.startswith("mac"):
        return "macos"
    return name


def _with_exe(path: Path) -> list[Path]:
    if path.suffix.lower() == ".exe":
        return [path]
    return [path, path.with_suffix(".exe")]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _configured_path(options: dict, keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        if value := options.get(key):
            return Path(str(value))
    return None


def _oscar_build_calibrator_candidates(oscar_root: Path, platform_name: str) -> list[Path]:
    candidates: list[Path] = []
    if platform_name == "windows":
        build_dirs = [
            "build-cuda/bin/Release",
            "build-cuda/bin",
            "build-cpu-avx2-nonative/bin/Release",
            "build-cpu-static/bin/Release",
            "build-win/bin/Release",
            "build/bin/Release",
        ]
    elif platform_name == "linux":
        build_dirs = [
            "build-cpu-avx2-nonative/bin",
            "build-cpu-static/bin",
            "build-cuda/bin",
            "build-linux/bin",
            "build/bin",
        ]
    else:
        build_dirs = ["build/bin", "build-cuda/bin"]

    for build_dir in build_dirs:
        candidates.extend(_with_exe(oscar_root / build_dir / CALIBRATOR_NAME))
    return candidates


def _llama_debug_candidates(root: Path, platform_name: str) -> list[Path]:
    if platform_name == "windows":
        return [
            root / "build-win" / "bin" / "Release" / "llama-debug.exe",
            root / "build" / "bin" / "Release" / "llama-debug.exe",
            root / "build" / "bin" / "llama-debug.exe",
            root / "build-cuda" / "bin" / "llama-debug.exe",
            root / "build-cpu-avx2-nonative" / "bin" / "Release" / "llama-debug.exe",
        ]
    if platform_name == "linux":
        return [
            root / "build-linux" / "bin" / "llama-debug",
            root / "build" / "bin" / "llama-debug",
            root / "build-cuda" / "bin" / "llama-debug",
            root / "build-cpu-avx2-nonative" / "bin" / "llama-debug",
        ]
    return [
        root / "build" / "bin" / "llama-debug",
        root / "build-cuda" / "bin" / "llama-debug",
    ]


def calibrator_candidates(options: dict, system: str | None = None) -> list[Path]:
    platform_name = _platform_name(system)
    candidates: list[Path] = []

    if requested := _configured_path(
        options,
        ("calibrator_bin", "llama_rot_kv_calibrate", "bin", "llama_debug"),
    ):
        candidates.extend(_with_exe(requested))

    if configured := options.get(f"llama_rot_kv_calibrate_{platform_name}"):
        candidates.extend(_with_exe(Path(str(configured))))

    by_platform = options.get("llama_rot_kv_calibrate_by_platform")
    if isinstance(by_platform, dict) and (configured := by_platform.get(platform_name)):
        candidates.extend(_with_exe(Path(str(configured))))

    oscar_roots: list[Path] = []
    if oscar_root := options.get("oscar_root"):
        oscar_roots.append(Path(str(oscar_root)))
    if DEFAULT_OSCAR_ROOT.is_dir():
        oscar_roots.append(DEFAULT_OSCAR_ROOT)

    for oscar_root in oscar_roots:
        candidates.extend(_oscar_build_calibrator_candidates(oscar_root, platform_name))
        candidates.extend(_llama_debug_candidates(oscar_root, platform_name))

    if root := options.get("llama_cpp_root"):
        llama_cpp_root = Path(str(root))
        candidates.extend(_oscar_build_calibrator_candidates(llama_cpp_root, platform_name))
        candidates.extend(_llama_debug_candidates(llama_cpp_root, platform_name))

    return _dedupe_paths(candidates)


def resolve_calibrator_bin(options: dict, system: str | None = None) -> str:
    for candidate in calibrator_candidates(options, system=system):
        if candidate.is_file():
            return str(candidate)

    platform_name = _platform_name(system)
    if configured := _configured_path(
        options,
        ("calibrator_bin", "llama_rot_kv_calibrate", "bin", "llama_debug"),
    ):
        if configured.is_file():
            return str(configured)
        raise MissingDependencyError(f"missing calibrator binary: {configured}")

    raise MissingDependencyError(
        "llama-rot-kv-calibrate (or llama-debug fallback) is required for OSCAR K/V rotation calibration. "
        "Build the OSCAR llama.cpp fork target `llama-rot-kv-calibrate`, then either:\n"
        "  - pass --bin to the dump script,\n"
        "  - set oscar_root to your OSCAR checkout build tree, or\n"
        f"  - place {CALIBRATOR_NAME} under third_party/OSCAR/build-cuda/bin/Release/."
    )


def calibrator_runtime_env(bin_path: str, env: dict[str, str] | None = None) -> dict[str, str]:
    runtime_env = dict(env or os.environ)
    paths = [str(Path(bin_path).resolve().parent)]
    if _platform_name() == "windows":
        cuda_root = Path(runtime_env.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"))
        paths.extend([str(cuda_root / "bin" / "x64"), str(cuda_root / "bin")])
    existing = runtime_env.get("PATH", "")
    runtime_env["PATH"] = os.pathsep.join([*paths, existing]) if existing else os.pathsep.join(paths)
    return runtime_env


def ensure_gguf_py_on_path() -> Path:
    if not GGUF_PY_ROOT.is_dir():
        raise MissingDependencyError(
            f"Bundled gguf-py is missing at {GGUF_PY_ROOT}. "
            "Run git submodule update --init --recursive."
        )
    return GGUF_PY_ROOT
