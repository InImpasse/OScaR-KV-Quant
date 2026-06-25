#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/OSCAR/gguf-py"))

from gguf import GGUFReader  # noqa: E402


def parse_meta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def load_logits(root: Path, variant: str) -> np.ndarray:
    tensor_dir = root / variant / "tensors"
    meta = parse_meta(tensor_dir / "result_output.0.meta.txt")
    if meta["type"] != "f32":
        raise ValueError(f"{variant}: expected f32 logits, got {meta['type']}")
    logits = np.fromfile(tensor_dir / "result_output.0.bin", dtype=np.float32)
    return logits


def field_value(reader: GGUFReader, name: str):
    field = reader.fields[name]
    parts = field.parts
    if len(parts) == 1:
        return parts[0]
    return parts


def load_vocab(model: Path) -> tuple[list[str], dict[str, int]]:
    reader = GGUFReader(str(model))
    tokens_raw = field_value(reader, "tokenizer.ggml.tokens")
    tokens = [bytes(t).decode("utf-8", errors="replace") for t in tokens_raw]
    ids = {}
    for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.padding_token_id"):
        if key in reader.fields:
            ids[key.rsplit(".", 1)[-1]] = int(field_value(reader, key)[0])
    return tokens, ids


def rank_of(logits: np.ndarray, token_id: int) -> int:
    return int(np.sum(logits > logits[token_id]) + 1)


def top_rows(variant: str, logits: np.ndarray, tokens: list[str], special: dict[str, int], topk: int) -> list[dict[str, str]]:
    order = np.argsort(logits)[-topk:][::-1]
    rows = []
    for rank, token_id in enumerate(order, start=1):
        special_name = ""
        for name, sid in special.items():
            if sid == int(token_id):
                special_name = name
                break
        rows.append({
            "variant": variant,
            "rank": str(rank),
            "token_id": str(int(token_id)),
            "logit": f"{float(logits[token_id]):.8g}",
            "piece": repr(tokens[int(token_id)] if int(token_id) < len(tokens) else ""),
            "special": special_name,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf"))
    parser.add_argument("--reference", default="oscar_bf16")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    tokens, special = load_vocab(args.model)
    variants = sorted(p.name for p in args.root.iterdir() if (p / "tensors/result_output.0.bin").exists())
    ref_logits = load_logits(args.root, args.reference)
    ref_top = int(np.argmax(ref_logits))

    rows = []
    summary = []
    for variant in variants:
        logits = load_logits(args.root, variant)
        rows.extend(top_rows(variant, logits, tokens, special, args.topk))
        eos_id = special.get("eos_token_id", -1)
        pad_id = special.get("padding_token_id", -1)
        summary.append({
            "variant": variant,
            "top_id": str(int(np.argmax(logits))),
            "top_piece": repr(tokens[int(np.argmax(logits))]),
            "top_logit": f"{float(np.max(logits)):.8g}",
            "ref_top_rank": str(rank_of(logits, ref_top)),
            "ref_top_logit": f"{float(logits[ref_top]):.8g}",
            "eos_rank": str(rank_of(logits, eos_id)) if eos_id >= 0 else "",
            "eos_logit": f"{float(logits[eos_id]):.8g}" if eos_id >= 0 else "",
            "pad_rank": str(rank_of(logits, pad_id)) if pad_id >= 0 else "",
            "pad_logit": f"{float(logits[pad_id]):.8g}" if pad_id >= 0 else "",
        })

    out_csv = args.root / "top_tokens.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = args.root / "top_token_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    out_md = args.root / "top_token_summary.md"
    with out_md.open("w") as f:
        f.write("# Top Token Summary\n\n")
        f.write("| variant | top piece | ref top rank | EOS rank | PAD rank |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in summary:
            f.write(f"| {r['variant']} | {r['top_piece']} | {r['ref_top_rank']} | {r['eos_rank']} | {r['pad_rank']} |\n")

    print(f"wrote {summary_csv}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
