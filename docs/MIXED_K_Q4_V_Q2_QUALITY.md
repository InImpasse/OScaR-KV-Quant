# Mixed K=q4, V=q2 Quality Variant

`oscar_kq4_vq2` is a llama.cpp quality-first mixed KV variant:

- model: rotated Granite GGUF
- K cache: `q4_0`
- V cache: `q2_0`
- q2 writer env: OSCAR no-Hadamard q2 path for V

It is not exact OSCAR INT2. It spends more KV memory on K to avoid the observed
q2 KQ drift, while keeping V at 2-bit. The purpose is to move llama.cpp quality
toward the external-reference level before returning to lower-bit K formats.

## Current Probe

Archive: `runs/mixed_k_quality_eval_current/`

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 1/3 | 1/3 | weak BF16 smoke baseline |
| oscar_int4 | 1/3 | 1/3 | q4/q4 control |
| oscar_int2 | 0/3 | 0/3 | exact q2/q2 failure |
| oscar_kq4_vq2 | 1/3 | 2/3 | restores normal answer shape and reaches INT4/BF16 band |
| plain_kq4_vq2 | 0/3 | 0/3 | rotation remains part of the quality path |

Direct prompt archive: `runs/mixed_k_quality_probe_current/`.
`oscar_kq4_vq2` answers the first arithmetic token correctly (`4`) and no longer
matches the exact q2/q2 immediate failure pattern, though it still drifts after
the first answer. This is a usable quality direction, not a final replacement
for a full accuracy benchmark.

## Next Step

The 10-sample follow-up shows `oscar_kq4_vq2` improves over exact q2/q2 but is
still a little below the BF16/INT4 band:

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 3/10 | 4/10 | BF16 smoke baseline |
| oscar_int4 | 4/10 | 4/10 | q4/q4 control |
| oscar_int2 | 0/10 | 0/10 | exact q2/q2 failure |
| oscar_kq4_vq2 | 2/10 | 3/10 | improves output shape but not fully in band |

Archive: `runs/mixed_k_quality_eval_10_current/`.

`oscar_kq4_vturbo3` is the stronger current quality candidate:

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 3/10 | 4/10 | BF16 smoke baseline |
| oscar_int4 | 4/10 | 4/10 | q4/q4 control |
| oscar_kq4_vq2 | 2/10 | 3/10 | q4/q2 mixed KV |
| oscar_kq4_vturbo3 | 4/10 | 4/10 | reaches INT4/BF16 band |
| plain_kq4_vturbo3 | 3/10 | 4/10 | rotation remains useful but not the only factor |

Archive: `runs/mixed_k_turbo3_quality_eval_10_current/`.

The direct CLI follow-up confirms that the already fast `oscar_int4` route is
also in the BF16 band:

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 3/10 | 4/10 | BF16 smoke baseline |
| oscar_int4 | 4/10 | 4/10 | reaches BF16 band |

Archive: `runs/oscar_int4_cli_quality_10_current/`.

Given the 32k speed/memory table below, `oscar_int4` is the current completed
llama.cpp target. `oscar_kq4_vturbo3` remains useful as a lower-memory research
direction, but its long-prefill speed is not product-ready.

## INT4 Delivery Refresh (2026-06-23)

Archive: `runs/oscar_int4_delivery_32k_20260623/`.

| variant | prompt | KV | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---|---:|---:|---:|---:|
| baseline_bf16 | 32768 | bf16/bf16 | 2560.0 | 6142 | 2586.6 | 49.4 |
| oscar_int4 | 32768 | q4_0/q4_0 | 720.0 | 4306 | 2576.8 | 36.8 |

The INT4 peak drop is `1836 MiB`, essentially matching the `1840 MiB` KV pool
drop, and prefill is `99.6%` of BF16 in the same run. Combined with the 10+10
quality smoke above, `oscar_int4` remains the completed llama.cpp route.

K3/V2 spike note: a minimal reuse attempt with K=`turbo3`, V=`q2_0` was compiled
as a D=128 FA vec probe, but even `p16/n1` exited with code 139 on GPU. The probe
was reverted from default dispatch/build/parser paths and should not be treated
as a usable candidate without a new kernel/layout audit.

## 32k Speed / Memory

Before the mixed FA instances were compiled into the default CUDA build, the
current quality candidates had good memory behavior but no valid 32k speed
result:

| variant | prompt | KV | status | peak MiB | note |
|---|---:|---|---|---:|---|
| oscar_kq4_vq2 | 32768 | q4_0/q2_0 | failed | 4041 | 240s timeout, empty JSON |
| oscar_kq4_vturbo3 | 32768 | q4_0/turbo3 | failed | 4081 | 240s timeout, empty JSON |

Archives:

- `runs/oscar_kq4_vq2_32k_current/`
- `runs/oscar_kq4_vturbo3_32k_current/`

This means the mixed q4/turbo3 path can match the BF16/INT4 smoke band by
raising K to q4 and V to turbo3, but its 32k prefill path still needs kernel
work. The completed target for now is `oscar_int4`: it matches the BF16 quality
band, has healthy 32k speed, and its peak VRAM drop tracks the KV cache drop.

## Mixed FA Instance Probe

The default CUDA FA build now compiles D=128 vector FA instances for:

- `q4_0/q2_0`
- `q4_0/turbo3`

`fattn.cu` also permits those two mixed pairs through the non-`FA_ALL_QUANTS`
support check. This is a llama.cpp-only change in the CUDA FA path.

For `oscar_kq4_vturbo3`, the first mixed-FA build used the existing turbo
`cols_per_block = 2` policy:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 532.3 | 42.8 | `runs/oscar_kq4_vturbo3_2k_after_mixed_fa_serial/` |
| 8192 | 180.0 | 3601 | 174.1 | 43.2 | `runs/oscar_kq4_vturbo3_8k_after_mixed_fa_serial/` |

Changing the mixed non-turbo-K/turbo-V vector FA case to
`cols_per_block = 4` improves prefill substantially:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 807.7 | 44.5 | `runs/oscar_kq4_vturbo3_2k_cols4_test/` |
| 8192 | 180.0 | 3601 | 279.4 | 49.5 | `runs/oscar_kq4_vturbo3_8k_cols4_test/` |
| 16384 | 360.0 | 3761 | 149.0 | 47.4 | `runs/oscar_kq4_vturbo3_16k_cols4_test/` |

Increasing turbo-V vector-lane participation from 4 lanes to 8 lanes per warp
adds another small improvement while preserving the same memory behavior:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 877.8 | 46.1 | `runs/oscar_kq4_vturbo3_2k_vthreads8_test/` |
| 8192 | 180.0 | 3601 | 306.2 | 47.6 | `runs/oscar_kq4_vturbo3_8k_vthreads8_test/` |
| 16384 | 360.0 | 3761 | 163.9 | 44.0 | `runs/oscar_kq4_vturbo3_16k_vthreads8_test/` |

An attempted `cols_per_block = 8` probe regressed at 2048 tokens:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3509 | 729.3 | 40.8 | `runs/oscar_kq4_vturbo3_2k_cols8_test/` |

An attempted intermediate `cols_per_block = 6` probe also failed to beat the
kept `cols_per_block = 4` policy and was reverted:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 872.4 | 38.1 | `runs/oscar_kq4_vturbo3_2k_ncols6_test/` |

Inlining the `turbo3` V half2 dequantization inside the vector FA accumulation
loop did not improve prefill and was reverted:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 877.0 | 46.3 | `runs/oscar_kq4_vturbo3_2k_inline_turbo3v_test/` |
| 8192 | 180.0 | 3601 | 306.2 | 36.8 | `runs/oscar_kq4_vturbo3_8k_inline_turbo3v_test/` |

Increasing turbo-V participation further to 16 lanes per warp also regressed
and was reverted:

| prompt | KV pool MiB | peak MiB | pp tok/s | tg tok/s | archive |
|---:|---:|---:|---:|---:|---|
| 2048 | 45.0 | 3485 | 863.4 | 10.7 | `runs/oscar_kq4_vturbo3_2k_vthreads16_test/` |

Conclusion: `cols_per_block = 4` plus 8 turbo-V lanes is the best tested policy
for the current mixed q4/turbo3 FA vector path. It is a real speedup over the
missing-instance state, but 16k is still only 163.9 tok/s, so 32k should remain
on hold until the kernel does more than reuse the existing vector template.

## Tile / MMA Boundary

The current CUDA dispatcher intentionally keeps q2/turbo KV types on vector FA.
This is not just a conservative policy: `fattn-tile.cuh` currently reads K and V
as `half2` pointers (`const half2 * K_h2`, `const half2 * V_h2`). Sending
`q4_0/turbo3` directly to tile or MMA by changing the enum selection would read
packed quantized KV bytes as fp16 data and corrupt attention.

The next real speed step is therefore a quant-aware tile load for the
`q4_0/turbo3` quality candidate: dequantize a KV tile into shared memory once,
reuse it across a larger query tile, and keep the same online-softmax semantics.
That is the CUDA analogue of the FutureMLS Metal tiled mixed-FA direction; it is
not a dispatch-only change.

A CUDA prototype interface now exists in `fattn-q4_0-turbo3.cu`, guarded by
`LLAMA_Q4_TURBO3_TILE_FA`, but it is intentionally disabled after timing worse
than the retained vector FA path. The correctness-first query-tiled q4-K/turbo3-V
kernel for D=128 reached only ~113 tok/s at pp256, and a local q8/dp4a variant
fell to ~76 tok/s, while the retained vector path was ~1474 tok/s on the same
pp256 smoke and ~910 tok/s at pp2048. The lesson is that a standalone scalar or
locally requantized prototype is not enough; the next viable step must either
extend the existing vector FA V path or implement a real quant-aware KV tile that
does not rebuild the optimized q4 KQ machinery from scratch.
