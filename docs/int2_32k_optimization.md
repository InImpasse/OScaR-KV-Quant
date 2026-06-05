# INT2 32K Optimization Notes

Date: 2026-06-04

## Problem

The 32K Granite benchmark originally showed INT2 using more peak GPU memory
than BF16 and running much slower:

| mode | prefill tokens | tok/s | peak MiB | KV pool tokens |
|---|---:|---:|---:|---:|
| bf16 | 32768 | 29.62 | 6572 | 38272 |
| int2 | 32768 | 13.55 | 7888 | 245632 |

The INT2 KV cache itself was smaller, but the implementation and benchmark
configuration hid that benefit.

## Root Causes

1. SGLang auto-sized the INT2 KV pool much larger than BF16.

   BF16 allocated 38272 KV tokens, while INT2 allocated 245632 KV tokens under
   the same `--mem-fraction-static=0.88`. This made INT2 reserve nearly all
   remaining GPU memory before the 32K request started.

2. The SM120 INT2 dense-prefill fallback materialized a large boolean causal
   mask.

   In `triton_backend.py`, the SM120 path falls back to
   `torch.nn.functional.scaled_dot_product_attention`. The previous code built
   an explicit `extend_len x k_len` boolean mask with `torch.arange`, which is
   costly at long context.

3. INT2 prefill still dequantizes prefix KV into dense tensors.

   `_forward_extend_quantized_dense` dequantizes cached INT2 prefix K/V, cats it
   with current chunk K/V, and then calls dense attention. This is still a
   structural limitation of the current implementation, but the two changes
   below reduce the worst observed overhead.

## Changes

### 1. Add `--max-total-tokens` to the benchmark wrapper

File: `src/oscar_kv_quant/bench.py`

The benchmark now accepts:

```bash
--max-total-tokens 38272
```

and forwards it to `sglang.launch_server` as:

```bash
--max-total-tokens 38272
```

This lets BF16 and INT2 allocate the same total KV pool capacity for fair memory
comparison.

### 2. Use PyTorch lower-right causal bias on SM120 INT2 prefill

File:
`third_party/OSCAR/sglang-research/python/sglang/srt/layers/attention/triton_backend.py`

The SM120 INT2 dense-prefill fallback now tries:

```python
from torch.nn.attention.bias import causal_lower_right
attn_mask = causal_lower_right(extend_len, k_len)
```

instead of always materializing:

```python
q_pos = torch.arange(extend_len, device=q3.device)[:, None]
k_pos = torch.arange(k_len, device=q3.device)[None, :]
attn_mask = k_pos <= (prefix_len + q_pos)
```

If `causal_lower_right` is unavailable, the code falls back to the old explicit
mask.

## Verification

Syntax check:

```bash
.venv-oscar-kv/bin/python -m py_compile \
  src/oscar_kv_quant/bench.py \
  third_party/OSCAR/sglang-research/python/sglang/srt/layers/attention/triton_backend.py
```

Optimized INT2 run:

```bash
./scripts/bench.sh \
  --profile granite \
  --modes int2 \
  --prefill-tokens 32768 \
  --max-new-tokens 64 \
  --max-total-tokens 38272 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --bench-requests 2 \
  --results-dir results/kv_32k_bf16_int2_optimized_20260604
```

BF16 comparison run:

```bash
./scripts/bench.sh \
  --profile granite \
  --modes bf16 \
  --prefill-tokens 32768 \
  --max-new-tokens 64 \
  --max-total-tokens 38272 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --bench-requests 2 \
  --port 31988 \
  --results-dir results/kv_32k_bf16_cap38272_retry_20260604
```

The BF16 retry used `--port 31988` to avoid a stale `dist-init` port conflict on
`32888`.

## Results

| mode | prefill tokens | tok/s | peak MiB | KV pool tokens | KV allocation |
|---|---:|---:|---:|---:|---|
| bf16 | 32768 | 31.41 | 6563 | 38272 | K 1.46 GB + V 1.46 GB |
| int2 optimized | 32768 | 26.88 | 5877 | 38272 | K 0.18 GB + V 0.18 GB |
| int2 original | 32768 | 13.55 | 7888 | 245632 | K 1.17 GB + V 1.17 GB |

The optimized INT2 run reduced peak GPU memory by 2011 MiB compared with the
original INT2 run and improved decode throughput from 13.55 tok/s to 26.88
tok/s. Compared with BF16 at the same 38272-token KV pool size, optimized INT2
used 686 MiB less peak GPU memory and reached about 86 percent of BF16 decode
throughput.

## Remaining Limitation

This is not a full kernel-level INT2 prefill solution. INT2 prefill still
dequantizes prefix KV into dense tensors and uses dense attention on SM120. A
true paged quantized prefill kernel would be needed to fully realize INT2's
theoretical memory and speed benefits at long context.
