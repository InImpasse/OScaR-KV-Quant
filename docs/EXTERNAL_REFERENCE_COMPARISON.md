# External Reference Comparison

This note compares the current llama.cpp-only evidence against the
user-provided external reference branch. It is a read-only comparison: no code
from that branch is imported into this repo.

## Current llama.cpp Results

Source:

- `runs/llamacpp_32k_kv_matrix_current/combined.csv`
- `runs/llamacpp_32k_kv_matrix_current/combined.md`

| variant | prompt | status | KV | KV MiB | peak MiB | pp tok/s | tg tok/s | note |
|---|---:|---|---|---:|---:|---:|---:|---|
| baseline_bf16 | 32768 | ok | bf16/bf16 | 2560.0 | 6160 | 2486.4 | 41.6 | final BF16 baseline |
| oscar_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2533.8 | 39.2 | peak drop matches KV drop |
| plain_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2265.0 | 41.0 | healthy |
| oscar_int2 | 32768 | failed | q2_0/q2_0 | 480.0 | 4036 |  |  | 480s timeout, empty JSON |

The current llama.cpp INT4 result is strong: BF16 to INT4 reduces theoretical
KV pool by 1840 MiB and observed peak VRAM by 1836 MiB.

The current llama.cpp INT2 result is not a valid 32k speed result. It records
good memory behavior, but the run timed out before producing usable
`llama-bench` JSON.

## External Reference Facts

The external branch reports a 32k Granite run with these key properties:

| variant | request tok/s | steady decode tok/s | peak delta MiB | KV theory GiB | note |
|---|---:|---:|---:|---:|---|
| BF16 | 29.66 | 29.28 | 6466 | 2.505 | capped token pool |
| plain INT2 | 27.08 | 24.74 | 5786 | 0.313 | capped token pool |
| OSCAR INT2 | 26.78 | 36.26 | 5826 | 0.494 | capped token pool |

The important optimization there was not CUDA graph. The main fixes were:

- cap the total token pool near the actual 32k request size;
- avoid materializing an oversized dense causal mask on SM120;
- accept that long prefill still dequantizes prefix KV into dense tensors.

That branch still described true quantized paged prefill as future work.

## Comparison Validity

The speed columns are not directly comparable:

- current repo: `llama-bench` prefill throughput (`pp tok/s`) plus short decode
  throughput (`tg tok/s`);
- external reference: request and steady decode throughput from a different
  runtime stack.

Memory trends are comparable at a high level, but the measurement harnesses are
different. The safest interpretation is directional: does peak VRAM fall roughly
with the KV pool, and does the run complete with usable speed?

## What Is Better Here

- The current llama.cpp INT4 path reaches valid 32k prefill and its peak VRAM
  drop almost exactly matches the KV pool drop.
- The result stays inside llama.cpp/OSCAR and does not depend on external
  runtime behavior.
- There is no oversized KV-pool symptom in the llama.cpp evidence. INT4 and q2
  KV pool sizes are already close to the expected cache-size reduction.

## What Is Still Worse Here

- Current llama.cpp does not yet have a valid 32k INT2 speed result.
- The exact q2_0 flash-attention path is much slower than q4 at long context.
- CUDA graph is unlikely to close this gap alone because the failed run is
  dominated by long q2 attention work rather than small launch overhead.

## Implication

To beat the external reference on 32k INT2 inside llama.cpp, the next real step
is q2 CUDA work, not another harness change. The likely path is profiler-guided
kernel work on `flash_attn_ext_vec<128,4,q2_0,q2_0>` or a different/approximate
KV format. Until then, INT4 is the product-facing 32k choice in this repo.
