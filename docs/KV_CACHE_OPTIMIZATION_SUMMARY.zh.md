# KV Cache Optimization Summary

This file keeps the historical path name for compatibility with older links, but
the content is now an English sanitized summary.

## Summary

The llama.cpp CUDA path can run quantized KV cache experiments and the repository
contains wrappers for throughput, memory, and quality checks. Low-bit KV cache
substantially reduces the theoretical KV allocation, but total peak memory only
falls when KV is a meaningful part of the whole runtime footprint.

Historical short and medium context sweeps showed:

- Granite 1B benefits visibly from low-bit KV memory savings at medium and long
  prompts.
- Some larger or differently shaped models can remain dominated by model weights,
  runtime workspaces, allocator behavior, or slower attention kernels.
- Exact q2 prefill is the main speed bottleneck; decode was closer to BF16 in
  early sweeps.
- High-precision sink/recent windows are useful for quality protection, but they
  are not the smallest-memory setting for short contexts.

## KV Storage Ratios

Approximate theoretical K+V storage relative to F16 KV:

| KV type | Relative F16 KV |
|---|---:|
| `q8_0` | 53.1% |
| `q4_0` | 28.1% |
| `q2_0` | 18.8% |

These ratios describe the KV cache itself. Whole-GPU peak memory also includes
weights, CUDA context, temporary buffers, graph capture, and allocator effects.

## Implemented Harness Capabilities

- CUDA benchmark wrappers for multiple models, prompt lengths, KV modes, and
  VRAM sampling.
- Per-run config files that record cache types, relevant `LLAMA_KV_*` variables,
  corpus metadata for PPL checks, and guard settings.
- PPL and benchmark gates for catching large quality regressions or invalid
  performance claims.
- Support for asymmetric K/V cache experiments through `KV_PAIRS=K/V,...`.

## Guardrails

- Do not interpret repository-derived PPL smoke runs as final model quality.
- Do not default a low-bit kernel change without a quality gate and a throughput
  comparison against BF16.
- Do not assume theoretical KV savings directly equal whole-GPU peak savings.
- Treat q2/INT2 long-prefill performance as a research path until a dedicated
  CUDA attention kernel changes the current speed class.

## Current Delivery Direction

For Granite 4.0 1B in this llama.cpp branch, INT4 is the practical delivery
route. Exact q2/INT2 remains useful for research and regression tracking, but it
is not the current 32K speed delivery path.
