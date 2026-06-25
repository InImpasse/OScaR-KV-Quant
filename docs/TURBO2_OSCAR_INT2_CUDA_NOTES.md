# Turbo2/OSCAR INT2 CUDA Notes

Date: 2026-06-13

Scope: llama.cpp / ggml only. Other serving stacks are external comparison points, not code directions.

## What Was Tested

Three q2 CUDA shortcuts were tested and then removed because they do not satisfy the speed target:

1. A D=128 q2_0/q2_0 query-tile proof kernel.
   - It reused K/V q2 dequant across multiple queries.
   - It lost the existing vector FA kernel's KV parallelism and was slower at 512/2k.

2. `fattn-vec` q2 prefill `cols_per_block=8`.
   - It preserved the existing vector FA structure.
   - It regressed 512 versus the existing `cols_per_block=4` path.

3. A zero-mean q2 KQ dot shortcut.
   - It removed the `m*usum` term and one `dp4a` behind an env-gated experiment.
   - Result: 512 improved from 1805.66 to 1933.60 tok/s, but 2048 regressed from 1039.91 to 869.17 tok/s.
   - This is not stable enough and is not exact q2_0, so it was removed.

## TurboQuant Reference Findings

The TurboQuant CUDA fork is useful because it moves INT2/INT3 KV toward an upstream-style ggml type instead of trying to make ordinary `q2_0` exact dot fast.

Relevant reference files in `/tmp/llama-cpp-turboquant`:

- `ggml/include/ggml.h`
  - Adds `GGML_TYPE_TURBO2_0`, `GGML_TYPE_TURBO3_0`, `GGML_TYPE_TURBO4_0`.
- `ggml/src/ggml-common.h`
  - `block_turbo2_0`: one 128-value block, one fp16 norm, 2-bit indices.
  - This avoids the per-32-value mean term used by this repo's `block_q2_0`.
- `ggml/src/ggml-turbo-quant.c`
  - CPU reference quantize/dequantize for turbo2/3/4.
- `ggml/src/ggml-cuda/turbo-quant.cuh`
  - CUDA centroids and helper routines.
- `ggml/src/ggml-cuda/set-rows.cu`
  - Runtime KV cache writes into turbo2/3/4 compressed rows.
- `ggml/src/ggml-cuda/fattn-common.cuh`
  - Turbo2/3/4 KQ dot and V dequant for vector flash attention.
- `ggml/src/ggml-cuda/fattn-vec.cuh`
  - Turbo-specific vector FA optimizations, including centroid LUTs for decode.
- `src/llama-graph.cpp`
  - Forward WHT on Q before attention and inverse WHT after attention when V is turbo.

Note on upstream llama.cpp/TurboQuant: the useful reference is not "plain q3_K KV cache".
The transferable design is the dedicated Turbo type family (`TURBO2_0`/`TURBO3_0`/`TURBO4_0`),
where KV cache rows use centroid indices plus per-block norm and have explicit CUDA FA support.
That maps much better to OSCAR INT2 than trying to accelerate exact `q2_0`, because exact `q2_0`
keeps the per-block mean term and therefore pays a heavier KQ path.

## Implemented Turbo2 CUDA Path

Current branch implements an opt-in `GGML_TYPE_TURBO2_0` in llama.cpp/ggml only:

- 128-value block layout: one fp16 `norm` plus 32 bytes of 2-bit centroid indices.
- CPU reference quantize/dequantize and ggml type traits.
- CUDA `set_rows` support for writing KV cache rows directly as Turbo2.
- CUDA vector flash-attention support for D=128 `turbo2/turbo2`.
- Turbo2 FA KQ uses per-query shared-memory centroid LUTs.
- Turbo2 FA currently uses `cols_per_block=2`; `cols_per_block=4` was tested and rejected.

The current implementation is intentionally still short of full TurboQuant:

- No WHT/OSCAR graph-side transform has been ported yet.
- Turbo3 has only a minimal CUDA/ggml KV-cache path; Turbo4 has not been added.
- No sparse-V path has been imported.
- Exact `GGML_TYPE_Q2_0` remains on the precise LUT baseline.

## Implemented Turbo3 CUDA Probe

Current branch also implements a minimal `GGML_TYPE_TURBO3_0` probe in llama.cpp/ggml only:

- 32-value block layout: one fp16 `norm`, 8 bytes of 2-bit low indices, and 4 sign/high-bit bytes.
- CPU reference quantize/dequantize and ggml type traits.
- CUDA `set_rows` support for writing KV cache rows directly as Turbo3.
- CUDA vector flash-attention support for D=128 `turbo3/turbo3`.
- Turbo3 FA KQ uses an 8-centroid per-query shared-memory LUT.

This is a speed/layout probe, not a full TurboQuant import:

- No graph-side WHT/inverse WHT path is active for Turbo3.
- No InnerQ or sparse-V path is active.
- No mixed Turbo2/Turbo3 or Turbo3/Turbo4 FA cases have been instantiated.

## Current CUDA A/B Results

Command shape:

```bash
third_party/OSCAR/build-cuda/bin/llama-bench \
  -m checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
  -p <prompt> -n 1 -r 1 -ngl 999 -fa 1 \
  --cache-type-k <type> --cache-type-v <type>
```

Final kept Turbo2 path: multi-query LUT with `cols_per_block=2`.

| prompt | turbo2/turbo2 pp tok/s | q2_0/q2_0 pp tok/s | q4_0/q4_0 pp tok/s | bf16/bf16 pp tok/s |
|---:|---:|---:|---:|---:|
| 2048 | 1718.62 | 1017.10 | 4800.62 | 4887.15 |
| 4096 | 1287.89 | 620.85 | 4477.14 | 4562.44 |
| 8192 | 756.45 | 333.13 | 4059.91 | 4047.72 |

Turbo2 now beats exact q2_0 by 1.7x-2.25x on 2k-8k, and 8k crosses the prior 700 tok/s
special-kernel gate. It is still far from q4/bf16 speed, so this is a useful step rather than
the final target.

Turbo3 probe A/B:

| prompt | turbo3/turbo3 pp tok/s | turbo2/turbo2 pp tok/s | q4_0/q4_0 pp tok/s |
|---:|---:|---:|---:|
| 512 | 1081.24 | 2492.04 | 3703.16 |
| 2048 | 542.95 | 1875.10 | 4758.29 |

Notes:

- Initial Turbo3 KQ without a centroid LUT was only 589.47 tok/s at 512.
- Adding the 8-centroid per-query LUT raised 512 to 1081.24 tok/s, but it is
  still far behind Turbo2 and q4.
- A parallel 2048 run was discarded because simultaneous GPU benchmarks
  interfere with each other; the table above uses sequential reruns.
- Decision: keep Turbo3 as a useful upstream/TurboQuant compatibility probe,
  but it is not the current path to beating q4/bf16. The 3-bit layout costs
  extra sign/high-bit decode and an 8-entry LUT, so it is not automatically
  easier than Turbo2 for CUDA prefill speed.

Kept Turbo2 V dequant fast path:

- Added a `ne == 4` fast path in CUDA `dequantize_V_turbo2_0` so the half2 V
  branch loads the block norm and packed byte once and emits two `half2`
  values directly.
- A/B:
  - 512: 2512.94 tok/s, essentially unchanged from 2513.54.
  - 2048: 1718.62 tok/s, essentially unchanged from 1719.24.
  - 8192: 756.45 tok/s, up from 748.65.
- Decision: keep. The gain is small, but the path is simpler and matches the
  TurboQuant CUDA dequant structure more closely.

Kept env-gated Turbo stream-k launch path:

- Idea: keep the Turbo2/Turbo3 vector FA kernel, but let it use llama.cpp's
  existing `launch_fattn(..., stream_k=true)` KV-axis partitioning and fixup
  instead of the default parallel-block reduce path.
- Runtime gate: `LLAMA_TURBO_VEC_STREAM_K=1`.
- This is different from the rejected tiled scaffold: it reuses the upstream
  CUDA FA launch/fixup framework instead of writing global partial tensors and
  a separate custom reduction kernel.
- Sequential A/B on RTX 5050:

| prompt | turbo2 stream-k pp tok/s | turbo2 default pp tok/s | q4_0/q4_0 pp tok/s |
|---:|---:|---:|---:|
| 512 | 4668.81 | 2363.51 | 3703.16 |
| 2048 | 5240.13 | 1769.54 | 4758.29 |
| 4096 | 5211.16 | — | 4685.21 |
| 8192 | 5041.24 | — | 4196.25 |
| 16384 | 4594.42 | — | 3395.11 |

- Decision: keep behind the env gate. This is the first Turbo2 CUDA path that
  beats q4_0/q4_0 through 16k in llama.cpp `llama-bench`. Next step is
  VRAM/KV-pool measurement and then the existing gated 32k matrix flow, not
  an ungated 32k q2 run.

16k VRAM/KV-pool measurement, same llama.cpp harness and single-case VRAM
sampler:

| variant | prompt | KV | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---|---:|---:|---:|---:|
| baseline_bf16 | 16384 | bf16/bf16 | 1280.0 | 4769 | 3342.0 | 54.4 |
| plain_int4 | 16384 | q4_0/q4_0 | 360.0 | 3825 | 3579.6 | 44.9 |
| turbo2_streamk | 16384 | turbo2/turbo2 | 170.0 | 3601 | 4943.4 | 59.0 |

Memory interpretation:

- BF16 -> q4_0: theoretical KV pool drops 920 MiB; measured peak drops
  944 MiB.
- q4_0 -> turbo2 stream-k: theoretical KV pool drops 190 MiB; measured peak
  drops 224 MiB.
- BF16 -> turbo2 stream-k: theoretical KV pool drops 1110 MiB; measured peak
  drops 1168 MiB.

This confirms that Turbo2 stream-k not only beats q4_0/q4_0 speed at 16k, but
also gives peak VRAM reduction close to the KV-cache storage reduction. Result
archive: `runs/turbo2_streamk_16k_vram_current/combined.md`.

Rejected Turbo2 `cols_per_block` A/B:

| prompt | ncols=1 pp tok/s | ncols=2 pp tok/s | ncols=3 pp tok/s | ncols=4 pp tok/s |
|---:|---:|---:|---:|---:|
| 512 | — | 2352.54 | 1906.33 | — |
| 2048 | 1696.56 | 1719.24 | — | 1129.70 |
| 4096 | 1115.24 | 1287.89 | — | 770.70 |
| 8192 | 635.34 | 748.65 | — | 420.15 |

Conclusion: ncols=2 is the best current tradeoff. ncols=4 over-pressures shared memory/registers
and ncols=3 already regresses at 512, so neither should be restored without a new kernel structure.

Rejected Turbo2 KQ register-forwarding A/B:

- Idea: for Turbo2 V aggregation, avoid writing `KQ_reg` to shared `KQ` and read it with
  `__shfl_sync`, matching a TurboQuant CUDA optimization.
- Result: 2048 was neutral (1720.59 vs 1719.24 tok/s), but 8192 regressed
  from 748.65 to 700.52 tok/s.
- Decision: reverted. The shared `KQ` handoff remains better on the RTX 5050 8k path.

Rejected Turbo LUT storage A/B:

- Idea: store Turbo2/Turbo3 per-query centroid LUT entries as `float` in shared
  memory instead of `half`, removing `__half2float` conversions in the KQ hot loop.
- Result:
  - Turbo2 512: 2473.10 tok/s, slightly below the half-LUT baseline around 2492-2513.
  - Turbo2 2048: 1781.95 tok/s, not better than the current 1718-1875 observed range.
  - Turbo3 512: 1075.97 tok/s versus 1081.24.
  - Turbo3 2048: 541.99 tok/s versus 542.95.
- Decision: reverted. The larger shared-memory footprint does not buy measurable
  speed, and it could hurt longer-context occupancy.

Rejected Turbo2 KQ norm-hoist A/B:

- Idea: in the Turbo2 KQ LUT loop, hoist the per-block `norm` load out of the
  inner 8-value chunk loop. For D=128 there is only one Turbo2 block per row.
- Result:
  - Turbo2 512: 2447.40 tok/s, below recent baseline runs around 2492-2513.
  - Turbo2 2048: 1722.66 tok/s, only baseline-level noise.
- Decision: reverted. This is another local vec-path cleanup that does not move
  toward the required q4/bf16 speed target.

Removed Turbo2 tiled CUDA scaffold:

- Tested an env-gated D=128 `turbo2/turbo2` prefill prototype behind
  `LLAMA_TURBO2_TILED_FA=1`.
- The first version processed 4 query rows per block and reused each decoded
  Turbo2 K/V row across those queries, but serially scanned the whole KV axis.
- First 512 A/B:
  - default Turbo2 vec path: 2513.54 tok/s
  - `LLAMA_TURBO2_TILED_FA=1`: 490.53 tok/s
- A second version split the KV axis into 128-token tiles and added a separate
  partial online-softmax reduction kernel, closer to FutureMLS Metal's
  Q-tile x KV-tile structure.
- Second 512 A/B:
  - default Turbo2 vec path: 2347.41 tok/s
  - `LLAMA_TURBO2_TILED_FA=1`: 1352.70 tok/s
- Decision: removed during final cleanup. Parallel KV tiling helped versus the
  serial prototype, but the global partial output and second reduction kernel
  still lost badly to the existing vector FA path at 512. The measured winning
  path reuses llama.cpp `launch_fattn` stream-k behind
  `LLAMA_TURBO_VEC_STREAM_K=1`.

## Current Conclusion

Continuing to optimize ordinary exact `GGML_TYPE_Q2_0` is unlikely to reach q4/bf16 speed:

- q2_0 KQ still needs sign/high LUT work, multiple `dp4a`, and mean handling.
- Removing the mean path improved only short 512 prefill and hurt 2k.
- Query-tile reuse without the existing FA parallelism also loses.

The next aligned implementation step is to finish the upstream-style INT2 KV cache route:

1. Port the WHT/OSCAR graph-side transform if accuracy/perplexity requires it.
2. Add a more fused tiled prefill kernel only if it keeps partial softmax state
   on-chip or otherwise avoids the global partial-output/reduce overhead shown
   by the current scaffold.
3. Run correctness/PPL checks before calling Turbo2 a replacement for OSCAR INT2.
4. Add VRAM peak/KV-pool measurement for Turbo2 at 16k and, only after the existing ACK gates, 32k.

Do not run 32k q2 until the existing gate ACK requirements are explicitly met.
