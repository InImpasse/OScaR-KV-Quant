#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("runs/q2_logits_path_dump_current")
WRITER_ROOT = Path("runs/q2_writer_ab_current")
IGNORE_EOS_ROOT = Path("runs/q2_ignore_eos_special_smoke_current")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def rows(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"missing {path}")
    return list(csv.DictReader(path.open()))


def find(rows_: list[dict[str, str]], variant: str) -> dict[str, str]:
    for r in rows_:
        if r["variant"] == variant:
            return r
    raise SystemExit(f"missing variant {variant}")


def read_bytes(name: str) -> bytes:
    path = IGNORE_EOS_ROOT / f"{name}.stdout.txt"
    require(path.exists(), f"missing {path}")
    return path.read_bytes()


def main() -> None:
    cur = rows(ROOT / "top_token_summary.csv")
    writer = rows(WRITER_ROOT / "top_token_summary.csv")

    bf16 = find(cur, "oscar_bf16")
    int4 = find(cur, "oscar_int4")
    int2 = find(cur, "oscar_int2")
    kq2 = find(cur, "oscar_kq2_vbf16")
    vq2 = find(cur, "oscar_kbf16_vq2")

    require(int(bf16["ref_top_rank"]) == 1, "BF16 reference top token must rank first")
    require(int(int4["ref_top_rank"]) == 1, "INT4 should preserve the BF16 top token")
    require(int(kq2["ref_top_rank"]) == 1, "K=q2 should preserve the BF16 top token in the current probe")
    require(int(int2["ref_top_rank"]) > 1, "INT2 should record top-token drift")
    require(int(vq2["ref_top_rank"]) > 1, "V=q2 should record top-token drift")
    require(int(int2["eos_rank"]) > 1000, "INT2 failure should not be explained by EOS rank explosion")
    require(int(vq2["eos_rank"]) > 1000, "V=q2 failure should not be explained by EOS rank explosion")

    split = find(writer, "q2_owht_split_clip_nohad")
    no_clip = find(writer, "q2_owht_no_clip_nohad")
    require(int(no_clip["ref_top_rank"]) > int(split["ref_top_rank"]),
            "no-clip improves logits cosine but not first-token rank in this probe")
    require(int(no_clip["eos_rank"]) > 1000 and int(split["eos_rank"]) > 1000,
            "writer A/B should also show EOS is not the main q2 failure")

    bf16_out = read_bytes("oscar_bf16")
    int4_out = read_bytes("oscar_int4")
    q2_owht = read_bytes("oscar_int2_owht_noclip")
    q2_plain = read_bytes("oscar_int2_plain")
    require(b"4" in bf16_out and b"4" in int4_out, "BF16/INT4 ignore-EOS smoke should answer 4")
    require(b"Solution: 2" in q2_owht, "OWHT q2 ignore-EOS smoke should show semantic drift, not immediate EOS-only failure")
    require(b"What is the" in q2_plain, "plain q2 ignore-EOS smoke should show semantic drift, not immediate EOS-only failure")

    print("q2 top-token drift checks passed")


if __name__ == "__main__":
    main()
