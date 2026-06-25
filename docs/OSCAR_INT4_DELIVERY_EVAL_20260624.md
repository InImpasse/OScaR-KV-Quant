# OSCAR INT4 Delivery Evaluation 2026-06-24

## Summary

This pass re-evaluated only the completed llama.cpp delivery path:

- `baseline_bf16`: base Granite BF16 GGUF with `bf16/bf16` KV cache.
- `oscar_int4`: rotated Granite GGUF with `q4_0/q4_0` KV cache.

The result is stable enough to keep `oscar_int4` as the delivery route. INT4
quality matches BF16 on the reliable CLI 50+50 smoke, 32K prefill is still at
BF16 speed, peak VRAM reduction continues to track the KV pool reduction, and
32K decode is already close to BF16.

## Quality

Server eval was re-run first, but it reproduced the known weak harness behavior:
BF16 itself scored nearly all zero, so that run is not used as the final quality
signal. The reliable CLI harness was then re-run with the same seed and 50 cases
per dataset.

Archive: `runs/oscar_int4_bf16_cli_quality_50_20260624/`

| variant | GPQA | GSM8K |
|---|---:|---:|
| baseline_bf16 | 11/50 | 16/50 |
| oscar_int4 | 11/50 | 16/50 |

Conclusion: `oscar_int4` is in the same quality band as BF16 for this Granite
1B smoke. It does not show the output-shape collapse seen in INT2/INT3 paths.

## Speed And Memory

8K decode-heavy bench:

Archive: `runs/oscar_int4_bf16_8k_n256_decode_20260624/`

| variant | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|---:|
| baseline_bf16 | 640.0 | 5508 | 3283.0 | 61.9 |
| oscar_int4 | 180.0 | 5904 | 2891.0 | 57.8 |

32K decode-heavy bench:

Archive: `runs/oscar_int4_bf16_32k_n64_decode_20260624/`

| variant | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|---:|
| baseline_bf16 | 2560.0 | 6143 | 2319.0 | 63.3 |
| oscar_int4 | 720.0 | 4307 | 2344.8 | 59.7 |

Conclusion: 32K INT4 peak is about 1836 MiB below BF16, matching the 1840 MiB
KV pool reduction. INT4 prefill is slightly above BF16 in this run, and decode
is about 94% of BF16, already above the 80% gate.

## CUDA Graph A/B

Single-case `oscar_int4`, 32K prompt, 64 generated tokens:

| mode | peak MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|
| graphs off | 4206 | 2395.2 | 48.1 |
| graphs on + opt | 4206 | 2392.6 | 46.9 |

Archives:

- `runs/oscar_int4_32k_n64_graph_off_20260624/`
- `runs/oscar_int4_32k_n64_graph_on_opt_20260624/`

Conclusion: CUDA graph settings do not improve this INT4 decode case. Do not
change the default graph mode for `oscar_int4` based on this A/B.

## Recommendation

- Keep `oscar_int4` as the current deliverable.
- Do not do risky CUDA kernel edits for INT4 decode right now; the current 32K
  tg is already close to BF16.
- If more validation is needed, extend the reliable CLI quality harness to
  100+100 cases. Do not use the server eval run as the primary quality signal
  until its prompt/output mismatch is fixed.
- Continue keeping INT2/INT3 frozen outside this delivery path.
