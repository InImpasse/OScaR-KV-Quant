# OSCAR llama.cpp KV Cache Validation

This repository is a local validation workspace for the
[`zhongzhu/llamacpp`](https://github.com/FutureMLS-Lab/OSCAR/tree/zhongzhu/llamacpp)
branch of [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR).

The focus is **llama.cpp + GGUF** experiments for KV-cache compression on a
local PC-class device.

## Repository Layout

- `third_party/OSCAR/`: llama.cpp fork with OSCAR/Q2_0 KV-cache support.
- `scripts/download_gguf_models.sh`: downloads trusted GGUF model files.
- `scripts/build_llamacpp.sh`: builds `llama-cli`, `llama-bench`, and helpers.
- `scripts/run_llamacpp.sh`: runs a single prompt with a selected KV cache type.
- `scripts/bench_kv_cache.sh`: compares KV-cache formats at fixed context size.
- `docs/`: llama.cpp validation notes and model source documentation.

## Trusted GGUF Models

The default experiments use BF16 GGUF weights so the model weights are not
another quantization variable. KV-cache compression is controlled at runtime by
`--cache-type-k` and `--cache-type-v`.

| Model | Source | Default file |
| --- | --- | --- |
| Gemma 4 E2B IT | `ggml-org/gemma-4-E2B-it-GGUF` | `gemma-4-E2B-it-bf16.gguf` |
| Granite 4.0 1B Base | `ibm-granite/granite-4.0-1b-base-GGUF` | `granite-4.0-1b-base-bf16.gguf` |

Download both models:

```bash
./scripts/download_gguf_models.sh
```

The files are placed under `~/models/gguf` by default. Override with:

```bash
GGUF_DIR=/path/to/models ./scripts/download_gguf_models.sh
```

## Build llama.cpp

```bash
./scripts/build_llamacpp.sh
```

CPU-only build:

```bash
LLAMACPP_CMAKE_ARGS="-DLLAMA_CURL=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF" \
  ./scripts/build_llamacpp.sh
```

CUDA build example:

```bash
LLAMACPP_CMAKE_ARGS="-DLLAMA_CURL=OFF -DGGML_CUDA=ON" \
  ./scripts/build_llamacpp.sh
```

## Run a Prompt

```bash
MODEL=~/models/gguf/granite-4.0-1b-base-bf16.gguf \
KV_TYPE=f16 \
CONTEXT=32768 \
PROMPT="Summarize why KV cache memory grows with context length." \
./scripts/run_llamacpp.sh
```

KV cache formats:

- `f16`: high-precision KV baseline.
- `q8_0`: 8-bit block-quantized KV cache.
- `q4_0`: 4-bit block-quantized KV cache.
- `q2_0`: 2-bit-ish KV cache type added by the OSCAR llama.cpp branch.

These runtime flags quantize **KV cache storage**, not model weights. A model
file such as `Q4_K_M.gguf` means the **weights** are quantized; that is a
separate variable.

## Benchmark KV Cache Formats

Run the same model and context length across several KV formats:

```bash
MODEL=~/models/gguf/granite-4.0-1b-base-bf16.gguf \
CONTEXT=32768 \
PROMPT_TOKENS=4096 \
GEN_TOKENS=512 \
./scripts/bench_kv_cache.sh
```

Results are written to `runs/llamacpp_kv_<timestamp>/`.

For an 8 GB GPU, Granite BF16 is the safer first target. Gemma E2B BF16 is much
larger and may require CPU execution or partial GPU offload.

## About OSCAR Calibration

The llama.cpp branch already ships Qwen3-4B-Thinking-2507 rotation files and a
GGUF baking script under `third_party/OSCAR/oscar-rotation/`. Those rotations are
model-specific.

**Granite:** this workspace’s OSCAR fork loads optional per-layer
`blk.{i}.attn_k_rot.weight` / `attn_v_rot.weight` for **granite** architectures
(post-RoPE, same graph semantics as `qwen3.cpp`). After you obtain
`k_rotation_qqt_r_h_pbr.pt` and `v_rotation_sst_r_h_pbr.pt` (same tensor layout
as the Qwen3 calibration — one `head_dim × head_dim` matrix per layer), bake them
with `third_party/OSCAR/oscar-rotation/export_rot_kv_gguf.py` into a
`*-rot-kv.gguf`. Producing those `.pt` files is **not** in this repo: upstream
OSCAR `README.md` describes GPQA activation dumps (e.g. via sglang) and the
**CoQuant `rotation/`** scripts for the `METHOD=qqt_sst` step; you need that
toolchain or an equivalent on your GPU.

**Gemma** and any model without the optional rotation tensors can still test KV
cache formats, but they are not calibrated OSCAR runs until matching rotations
exist and the model graph applies them.
