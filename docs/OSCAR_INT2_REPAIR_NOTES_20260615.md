# OSCAR INT2 Repair Notes (2026-06-15)

## Scope

This note records the latest attempt to repair `oscar_int2` for Granite 4.0 1B in the llama.cpp/OSCAR path.

The target variant is:

```bash
-m checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
-fa on -ctk q2_0 -ctv q2_0 \
LLAMA_KV_Q2_0_OWHT=1 LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0
```

## Negative CUDA Reader Experiment

An experimental q2/q2 OWHT-aware flash-attention reader was tried before this note. It explicitly dequantized each q2 row, optionally applied OWHT inverse, and then ran a correctness-first attention loop.

Result on the direct prompt:

```text
Question: What is 2 + 2? Answer with one number.
```

The output was still wrong/repetitive, and the path was substantially slower than the existing vector FA path. The experimental dispatch and kernel were removed from the tree.

Conclusion: the main issue is not a missing OWHT inverse in the CUDA q2/q2 reader.

## q2 Centroid Search

Added:

```bash
scripts/search_q2_centroids.py
scripts/search_q2_kv_split_centroids.py
```

These are offline scripts over the existing Granite Q/K/V dump. They test whether `block_q2_0` can be rescued by changing the symmetric 2-bit reconstruction levels and thresholds while preserving the q2 layout.

Single K/V-shared best on 24 small attention cases:

```text
c0=0.720000, c1=0.200000, threshold=0.830000
mean attention NMSE: 0.646301
```

Original Lloyd-Max on the same sample:

```text
c0=0.981600, c1=0.452800, threshold=0.674500
mean attention NMSE: 1.069297
```

So tuning q2 centroids does improve offline attention error by about 40%, but K-only error remains high:

```text
best K-only mean NMSE: 0.990114
```

K/V-split centroid search did not improve over the shared best:

```text
best split mean attention NMSE: 0.708323
```

## Runtime Smoke

A temporary build-time q2 profile with the best shared centroid candidate was tested in an isolated build directory:

```bash
third_party/OSCAR/build-cuda-q2granite
```

Direct prompt result improved from the original q2 failure mode, but still failed:

```text
2 + 2 + 2 + 2 + 2 + ...
```

Small CLI quality smoke still failed:

```text
oscar_int2 gpqa  0/3
oscar_int2 gsm8k 0/3
```

The temporary profile was not kept in the core library.

## Current Conclusion

Exact `q2_0/q2_0` is still not repaired.

The evidence now rules out the following as sufficient fixes:

- CUDA graph launch optimization
- q2/q2 OWHT-aware CUDA read path
- split clipping
- HP sink/recent/full-context preservation
- global q2 centroid/threshold tuning
- K/V-split q2 centroid/threshold tuning

The remaining viable repair route is a new OSCAR-specific low-bit KV format rather than reusing `GGML_TYPE_Q2_0`. The new format must reduce K-side attention-score error; otherwise softmax amplifies the loss before V can matter.

## Recommended Next Implementation

Implement a new KV cache type, for example `GGML_TYPE_OSCAR2_KV`, with:

- K-side format optimized for dot products, not reconstruction NMSE only
- separate K/V quantization profiles
- group size 128 to match OSCAR/FutureMLS tiled mixed-FA assumptions
- CUDA q2-like vector FA dispatch from day one
- quality gate before speed work:
  - direct arithmetic prompt must answer `4`
  - GPQA/GSM8K 3+3 smoke must be non-zero
  - 10-case smoke should be in the same rough band as `oscar_int4`

Do not spend more time trying to tune `q2_0` constants unless new calibration evidence shows K-only attention NMSE can drop far below the current ~0.99 floor.
