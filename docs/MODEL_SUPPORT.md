# Model and KV Path Support Notes

These notes are based on a static audit of the bundled submodule `third_party/OSCAR/sglang-research`.

## SGLang Model Implementations

| Model | Hugging Face ID | Relevant implementation under `sglang/srt/models` |
|------|------------|--------------------------------|
| Granite 4.0 1B | `ibm-granite/granite-4.0-1b-base` | `granitemoehybrid.py` |
| Gemma 4 E2B | `google/gemma-4-E2B` | `gemma4_causal.py` as part of the multimodal stack |

Both models have upstream SGLang model files, but that does **not** guarantee the full INT2 + mixed KV + FA3 path works on RTX 5050 with the current dependency set. Treat `oscar-kv-probe` and actual `sglang.launch_server` runs as the final source of truth.

## KV Dtype Mapping

The bundled `server_args.py` accepts these `--kv-cache-dtype` values:

`auto`, `bf16`, `bfloat16`, `fp8_e5m2`, `fp8_e4m3`, `fp4_e2m1`, `int2`, **`int8`**, **`int4`**

- **`int8` / `int4` (this fork):** symmetric integer storage in the KV pool plus an internal **bf16 shadow** copy so existing Triton extend/decode kernels keep reading bf16 K/V. This is true integer quantization (not `fp8_e4m3` / `fp4_e2m1`). **Memory:** roughly int8 physical + 2× bf16 shadow per element (see `pool_configurator._mha_int_kv_bytes_per_head_pair`). **`int4`** requires **even** `head_dim` and `v_head_dim` (nibble packing along the last dimension). Speculative **CUDA-graph KV copy** is disabled for these dtypes until the copy kernels include the shadow buffers.

This repo’s benchmark modes map as follows:

| User-facing mode | SGLang `--kv-cache-dtype` | Notes |
|----------|-------------------------|------|
| `bf16` | `bf16` | BF16 KV-cache baseline |
| `fp8` | `fp8_e4m3` | 8-bit **floating** KV cache |
| `int8` | `int8` | Symmetric **integer** int8 KV + bf16 shadow (Triton) |
| `fp4` | `fp4_e2m1` | MXFP4 path; hardware / PyTorch dependent |
| `int4` | `int4` | Symmetric **integer** int4 (packed) + bf16 shadow (Triton) |
| `int2` | `int2` | SGLang Triton INT2 KV path without OSCAR rotations |
| `oscar-int2` | `int2` | OSCAR mixed KV windows plus rotation files |
| `oscar-int8` | `int8` | `int8` KV + `SGLANG_OSCAR_ROTATE_QUANT_KV` + rotation checkpoints |
| `oscar-int4` | `int4` | `int4` KV + `SGLANG_OSCAR_ROTATE_QUANT_KV` + rotation checkpoints |

## OSCAR INT2 With Rotations

The paper-style serving path requires:

- `--kv-cache-dtype int2`
- `--kv-cache-quant-group-size`, commonly `128`
- `SGLANG_ENABLE_MIXED_KV_WINDOWS`
- `SGLANG_OSCAR_K_ROTATION_PATH`
- `SGLANG_OSCAR_V_ROTATION_PATH`
- other OSCAR environment variables used in `third_party/OSCAR/rotation/eval_oscar_gpqa.sh`

**Mixed KV and hybrid SWA:** `server_args._unified_mixed_kv_active` disables the unified mixed KV path when `model_config.is_hybrid_swa` is true. If Gemma4 is marked as hybrid SWA, OSCAR-style mixed INT2 may not activate. Use runtime logs and `oscar-kv-probe` to confirm.

## Granite 4.0 1B

- The local `config.json` has `num_hidden_layers=40`, `num_attention_heads=16`, `num_key_value_heads=4`, and `hidden_size=2048`, so `head_dim = 128`.
- The architecture is **GraniteMoeHybrid**, with hybrid Mamba / attention behavior. The benchmark estimator prefers the number of attention entries in `layer_types`, but rotation compatibility must still be validated against the actual dump directory layout.

## Gemma 4 E2B

- Use `google/gemma-4-E2B`.
- Layer count and `head_dim` are resolved from `config.json` or nested `text_config` when present.
- `oscar-kv-bench` supports `--num-layers`, `--num-kv-heads`, and `--head-dim` overrides.
- Long contexts may OOM on 8GB VRAM. Start with `--preset short` or a smaller `--prefill-tokens`.

## Key Benchmark CLI Options

- `--request-api completions|chat|generate`: `completions` is recommended for base models.
- `--dry-run`: print the SGLang launch command, OSCAR environment variables, and theoretical KV estimates without starting a server.
- `--num-layers`, `--num-kv-heads`, `--head-dim`: override KV geometry parsed from config.
- `--prefix-bf16-tokens`, `--recent-bf16-tokens`: affect OSCAR mixed KV theoretical estimates.
- `--prefill-attention-backend`, `--decode-attention-backend`: override default backends, useful for isolating `fa3` / `triton` compatibility on RTX 5050.
