# llama.cpp Validation Notes

## Goals

This workspace validates KV-cache storage formats with the OSCAR llama.cpp fork:

- Keep model weights fixed, preferably BF16 GGUF.
- Change only `--cache-type-k` and `--cache-type-v`.
- Use the same model, prompt length, generation length, context length, and
  offload settings for every KV type.

## KV Cache Types

```bash
--cache-type-k f16  --cache-type-v f16
--cache-type-k q8_0 --cache-type-v q8_0
--cache-type-k q4_0 --cache-type-v q4_0
--cache-type-k q2_0 --cache-type-v q2_0
```

`q4_0` and `q2_0` here are KV-cache formats. They do not imply that the model
weights are quantized.

## Suggested First Experiment

Use Granite BF16 first on an 8 GB GPU:

```bash
MODEL=~/models/gguf/granite-4.0-1b-base-bf16.gguf \
CONTEXT=32768 \
PROMPT_TOKENS=4096 \
GEN_TOKENS=512 \
KV_TYPES=f16,q8_0,q4_0,q2_0 \
./scripts/bench_kv_cache.sh
```

If the GPU runs out of memory, lower one variable at a time:

- reduce `CONTEXT` to `16384`;
- reduce `PROMPT_TOKENS`;
- reduce GPU offload with `N_GPU_LAYERS`.

## OSCAR Rotation Caveat

The bundled calibrated rotation files are for Qwen3-4B-Thinking-2507. For Gemma
or Granite, `q2_0` tests the storage format and implementation path, but not a
full calibrated OSCAR result.
