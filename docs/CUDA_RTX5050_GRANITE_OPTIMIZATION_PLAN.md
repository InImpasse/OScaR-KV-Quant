# CUDA Optimization Plan: RTX 5050 Class GPU + Granite 4.0 1B BF16

This note tracks the historical llama.cpp CUDA validation plan for Granite 4.0
1B BF16 on a small local GPU. It is retained as a sanitized public reference and
uses only repository-relative paths.

## Scope

- Runtime: the vendored llama.cpp fork under `third_party/OSCAR/`.
- Model: `checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`.
- Primary tools: `llama-bench`, `llama-server`, and the shell wrappers in
  `scripts/`.
- KV modes for early sweeps: `f16`, `q8_0`, `q4_0`, and `q2_0`.

Granite BF16 without baked rotations is useful for KV storage and performance
validation. It is not a paper-level OSCAR INT2 accuracy reproduction by itself.

## Build Smoke

Configure and build the llama.cpp fork from the repository root:

```bash
cmake -S third_party/OSCAR -B third_party/OSCAR/build-cuda \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench llama-server
```

Check that the build sees the GPU:

```bash
third_party/OSCAR/build-cuda/bin/llama-bench --list-devices
```

## KV Sweep

Start with a conservative context and increase only one variable at a time:

```bash
OUT_DIR=runs/kv_smoke_$(date +%Y%m%d_%H%M%S) \
MODELS=granite:checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
LENGTHS=short:512,medium:2048,long:4096 \
KV_MODES=f16,q8_0,q4_0,q2_0 \
GEN_TOKENS=64 \
DRY_RUN=0 \
  scripts/bench_kv_cache_matrix.sh
```

Record:

- Prompt processing throughput (`pp`).
- Decode throughput (`tg`).
- Peak GPU memory from the harness logs or `nvidia-smi`.
- Whether flash attention was enabled.

Expected storage trend is `f16` > `q8_0` > `q4_0` > `q2_0` for KV memory. Speed
does not have to improve monotonically, because low-bit KV can shift bottlenecks
into dequantization and attention kernels.

## Current Direction

INT4 is the current successful llama.cpp delivery path for Granite. Exact q2 /
INT2 remains a research path and should be treated as guarded until a dedicated
D=128 tiled CUDA prefill kernel or equivalent kernel-level change lands.
