# GGUF Model Sources

Use trusted model publishers for the local GGUF files.

## Gemma 4 E2B IT

Source:

https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF

Recommended BF16 file:

```text
gemma-4-E2B-it-bf16.gguf
```

Download:

```bash
huggingface-cli download ggml-org/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-bf16.gguf \
  --local-dir ~/models/gguf
```

## Granite 4.0 1B Base

Source:

https://huggingface.co/ibm-granite/granite-4.0-1b-base-GGUF

Recommended BF16 file:

```text
granite-4.0-1b-base-bf16.gguf
```

Download:

```bash
huggingface-cli download ibm-granite/granite-4.0-1b-base-GGUF \
  granite-4.0-1b-base-bf16.gguf \
  --local-dir ~/models/gguf
```

## Weight Quantization vs KV Cache Quantization

Files such as `Q4_K_M.gguf` use quantized model weights. Runtime options such as
`--cache-type-k q4_0 --cache-type-v q4_0` quantize KV-cache storage. Keep those
two concepts separate when comparing memory and throughput.
