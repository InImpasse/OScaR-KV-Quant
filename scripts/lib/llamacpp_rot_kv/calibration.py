from __future__ import annotations

import json
from pathlib import Path

FIELD_CANDIDATES = ("prompt", "text", "question", "input")


def read_prompts(path: Path, max_prompts: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")

    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            prompt = raw
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
                for field in FIELD_CANDIDATES:
                    value = obj.get(field)
                    if isinstance(value, str) and value.strip():
                        prompt = value.strip()
                        break
                else:
                    raise ValueError(
                        f"{path}:{line_no}: JSONL row must contain one of {FIELD_CANDIDATES}"
                    )
            prompts.append(prompt)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts
