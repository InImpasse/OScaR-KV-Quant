# OSCAR2 KV Type Status (2026-06-15)

## What changed upstream

The llama.cpp/OSCAR submodule now has an experimental dedicated OSCAR INT2 KV cache type:

```text
GGML_TYPE_OSCAR2_KV -> cache type name oscar2
```

This is intentionally separate from `GGML_TYPE_Q2_0`, so the existing exact q2 baseline remains available and the OSCAR-specific path can evolve without mutating q2 semantics. K and V share the public storage type but use role-specific quantization/dequantization profiles in the KV write path and CUDA FA path.

Implemented pieces:

- ggml type registration and type sizes
- CPU reference quant/dequant
- CLI and `llama-bench` cache type parsing
- CUDA `SET_ROWS`, selecting the V quantizer for `cache_v` tensors and the K quantizer otherwise
- CUDA vector flash-attention dispatch for `oscar2/oscar2`
- CUDA FA template instance for D=64/128/256/512

## Smoke result

Command:

```bash
LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0 \
third_party/OSCAR/build-cuda/bin/llama-bench \
  -m checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
  -p 512 -n 1 -r 1 -ngl 999 -fa 1 \
  --cache-type-k oscar2 --cache-type-v oscar2
```

Result:

```text
pp512: 865.06 tok/s
tg1:    67.62 tok/s
```

This confirms the type, CUDA set_rows, and CUDA FA path execute end-to-end.

## Quality status

The current `oscar2` quantizer uses separate scalar 4-level K/V codebooks from the previous offline search.

It is **not repaired yet**:

- direct prompt still repeats `2. 2. 2...`
- `oscar2_int2` GPQA/GSM8K 3+3 smoke is still `0/3 + 0/3`

Additional offline checks show that:

- FutureMLS-style affine/min-max q2 is worse on the Granite rotation dump
- free scalar 4-level codebook search does not beat the current failure floor

## Next required optimization

The engineering path is now in upstream OSCAR, not the outer harness. The next quality work should change the `oscar2` format itself, not `q2_0`.

Likely directions:

- K-side vector/group codebook optimized for KQ error, not scalar reconstruction NMSE
- small HP K tier plus INT2 V tier under one joint online softmax
- FutureMLS-style tiled mixed prefill kernel once the quality format passes smoke

Do not keep tuning scalar `q2_0`/`oscar2` centroids unless a new offline metric shows K-only attention error dropping substantially below the current floor.
