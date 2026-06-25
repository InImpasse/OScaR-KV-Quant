#!/usr/bin/env python3
import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-common.cuh"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def extract_function(text: str, name: str) -> str:
    start = text.find(name)
    if start < 0:
        raise AssertionError(f"missing function: {name}")
    brace = text.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing function body: {name}")
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise AssertionError(f"unterminated function body: {name}")


def fingerprint(body: str) -> str:
    normalized = "\n".join(line.rstrip() for line in body.strip().splitlines()) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


def rows_from_source(text: str) -> list[dict[str, str]]:
    q2_chunk = extract_function(text, "vec_dot_fattn_vec_KQ_q2_0_chunk")
    q2_kq = extract_function(text, "vec_dot_fattn_vec_KQ_q2_0")
    q4_kq = extract_function(text, "vec_dot_fattn_vec_KQ_q4_0")
    q2_v = extract_function(text, "dequantize_V_q2_0")
    q4_v = extract_function(text, "dequantize_V_q4_0")
    dispatch_kq = extract_function(text, "get_vec_dot_KQ")
    dispatch_v = extract_function(text, "get_dequantize_V")

    rows = [
        {
            "path": "KQ",
            "type": "q2_0",
            "function": "vec_dot_fattn_vec_KQ_q2_0_chunk",
            "dp4a_calls": str(q2_chunk.count("ggml_cuda_dp4a")),
            "lut": "sign+high" if "Q2_0_FATTN_SIGN_LUT" in q2_chunk and "Q2_0_FATTN_HIGH_LUT" in q2_chunk else "missing",
            "mean_term": "m*usum" if "m*usum" in q2_chunk else "missing",
            "scalar_decode": "no",
            "dispatch": "yes" if "type_K == GGML_TYPE_Q2_0" in dispatch_kq and "vec_dot_fattn_vec_KQ_q2_0" in dispatch_kq else "missing",
            "fingerprint": fingerprint(q2_chunk),
            "note": "exact q2 reconstruction",
        },
        {
            "path": "KQ",
            "type": "q4_0",
            "function": "vec_dot_fattn_vec_KQ_q4_0",
            "dp4a_calls": str(q4_kq.count("ggml_cuda_dp4a")),
            "lut": "no",
            "mean_term": "unexpected" if "m*usum" in q4_kq else "no",
            "scalar_decode": "no",
            "dispatch": "yes" if "type_K == GGML_TYPE_Q4_0" in dispatch_kq and "vec_dot_fattn_vec_KQ_q4_0" in dispatch_kq else "missing",
            "fingerprint": fingerprint(q4_kq),
            "note": "q4 comparison path",
        },
        {
            "path": "V",
            "type": "q2_0",
            "function": "dequantize_V_q2_0",
            "dp4a_calls": str(q2_v.count("ggml_cuda_dp4a")),
            "lut": "no",
            "mean_term": "no",
            "scalar_decode": "yes" if "q2_0_dequantize_scalar_cuda" in q2_v else "missing",
            "dispatch": "yes" if "type_V == GGML_TYPE_Q2_0" in dispatch_v and "dequantize_V_q2_0" in dispatch_v else "missing",
            "fingerprint": fingerprint(q2_v),
            "note": "scalar q2 dequant per V lane",
        },
        {
            "path": "V",
            "type": "q4_0",
            "function": "dequantize_V_q4_0",
            "dp4a_calls": str(q4_v.count("ggml_cuda_dp4a")),
            "lut": "no",
            "mean_term": "no",
            "scalar_decode": "no",
            "dispatch": "yes" if "type_V == GGML_TYPE_Q4_0" in dispatch_v and "dequantize_V_q4_0" in dispatch_v else "missing",
            "fingerprint": fingerprint(q4_v),
            "note": "packed nibble dequant",
        },
        {
            "path": "dispatch",
            "type": "KQ",
            "function": "get_vec_dot_KQ",
            "dp4a_calls": "0",
            "lut": "n/a",
            "mean_term": "n/a",
            "scalar_decode": "n/a",
            "dispatch": "yes",
            "fingerprint": fingerprint(dispatch_kq),
            "note": "KQ type dispatch",
        },
        {
            "path": "dispatch",
            "type": "V",
            "function": "get_dequantize_V",
            "dp4a_calls": "0",
            "lut": "n/a",
            "mean_term": "n/a",
            "scalar_decode": "n/a",
            "dispatch": "yes",
            "fingerprint": fingerprint(dispatch_v),
            "note": "V type dispatch",
        },
    ]

    q2_kq_uses_chunk = "vec_dot_fattn_vec_KQ_q2_0_chunk" in q2_kq
    require(q2_kq_uses_chunk, "q2 KQ wrapper should call q2 chunk path")
    require(rows[0]["dp4a_calls"] == "3", "q2 KQ chunk must use 3 dp4a calls")
    require(rows[1]["dp4a_calls"] == "1", "q4 KQ must use 1 dp4a call")
    require(rows[0]["mean_term"] == "m*usum", "q2 KQ must include exact mean term")
    require(rows[2]["scalar_decode"] == "yes", "q2 V must use scalar decode path")
    for row in rows:
        require(row["dispatch"] == "yes", f"{row['type']} {row['path']} dispatch missing")
    return rows


def fields() -> list[str]:
    return ["path", "type", "function", "dp4a_calls", "lut", "mean_term", "scalar_decode", "dispatch", "fingerprint", "note"]


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields())
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fields()) + " |",
        "|" + "|".join("---" for _ in fields()) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields()) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report static q2/q4 CUDA attention path facts without running GPU code.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    text = args.source.read_text(errors="replace")
    rows = rows_from_source(text)
    for row in rows:
        print(
            f"{row['path']} {row['type']}: function={row['function']} dp4a={row['dp4a_calls']} "
            f"lut={row['lut']} mean={row['mean_term']} scalar_decode={row['scalar_decode']} "
            f"dispatch={row['dispatch']} fingerprint={row['fingerprint'][:12]}"
        )

    if args.out_dir is not None:
        write_csv(rows, args.out_dir / "q2_cuda_path.csv")
        write_md(rows, args.out_dir / "q2_cuda_path.md")
        print(f"wrote {args.out_dir / 'q2_cuda_path.csv'}")
        print(f"wrote {args.out_dir / 'q2_cuda_path.md'}")


if __name__ == "__main__":
    main()
