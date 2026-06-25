#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ARCHIVES = {
    Path("runs/cuda_graph_ab_512_current"),
    Path("runs/goal_status_current"),
    Path("runs/llamacpp_32k_kv_matrix_current"),
    Path("runs/q2_cuda_path_current"),
}


def archive_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{path} is outside repo root") from exc
    else:
        rel = path

    rel = Path(*rel.parts)
    if rel not in ALLOWED_ARCHIVES:
        allowed = ", ".join(str(p) for p in sorted(ALLOWED_ARCHIVES))
        raise argparse.ArgumentTypeError(f"expected one of: {allowed}")
    return ROOT / rel


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh SHA256SUMS for checked-in run archives using repo-root-relative paths."
    )
    parser.add_argument("archive", type=archive_path)
    args = parser.parse_args()

    archive = args.archive
    files = [
        path
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    if not files:
        raise SystemExit(f"no archive files found under {archive.relative_to(ROOT)}")

    out = archive / "SHA256SUMS"
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(files)} files)")


if __name__ == "__main__":
    main()
