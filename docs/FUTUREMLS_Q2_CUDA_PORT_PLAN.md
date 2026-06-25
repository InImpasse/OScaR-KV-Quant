# FutureMLS Q2 CUDA Port Plan

This repo stays on the llama.cpp / ggml path. External server-runtime results are
reference numbers only, not an implementation direction.

## Reference Branch

Reference inspected:

- `FutureMLS-Lab/OSCAR`, branch `zhongzhu/llamacpp`
- fetched in `third_party/OSCAR` as `futuremls/zhongzhu/llamacpp`
- reference commit observed locally: `21899d6b407e3a49e55c11cf1f228bf97ffb4c4b`

## Important Finding

The useful FutureMLS performance idea is the Metal tiled mixed-precision prefill
kernel, not the existing local CUDA q2 helper.

FutureMLS Metal path:

- `ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - detects OSCAR mixed two-tier attention with `op_params[4] == 1`
  - dispatches `kernel_flash_attn_mixed_mm_q2_0_f16_d128` for prefill
  - uses `Q=8` query rows and `C=64` KV columns per threadgroup
  - dequantizes q2 K/V once per KV tile and reuses the tile across the Q rows
  - keeps one online softmax state across LP q2 and HP f16 tiers
- `ggml/src/ggml-metal/ggml-metal.metal`
  - `kernel_flash_attn_mixed_mm`
  - `mm_mixed_pass`
  - `dequantize_q2_0`

Current local CUDA q2+f16 helper:

- `ggml/src/ggml-cuda/fattn-q2_0-f16.cu`
- one block per token/head
- loops over every KV position and every head dimension
- dequantizes q2 with scalar loads in the inner loop
- useful as a correctness scaffold, not a performance path

Current exact q2/q2 CUDA vec path:

- `ggml/src/ggml-cuda/fattn-common.cuh`
- q2 KQ uses sign/high LUT plus multiple `dp4a` operations and `m*usum`
- q4 KQ is much cheaper, so small local tweaks are unlikely to close the 32k gap

## CUDA Implementation Direction

Do not spend more effort on small changes inside the current q2 vec dot path.
The next CUDA work should introduce a dedicated D=128 q2/q2 prefill kernel that
ports the FutureMLS tiled idea:

- query tile: 8 tokens
- KV tile: 64 positions
- head dim: initially 128 only
- K/V type: q2_0/q2_0 exact decode
- softmax: online, one pass over tiled K blocks
- V accumulation: tile probabilities times q2 V decoded tile
- dispatch: guarded by an env flag and shape/type checks

The initial kernel can ignore mixed HP f16 and target the user-requested 32k
q2_0/q2_0 llama.cpp benchmark first. HP two-tier support can follow after the
q2/q2 tiled kernel is correct and faster than the vec path.

## CUDA Skeleton Status (2026-06-16)

Implemented in-tree (default off):

- `ggml/src/ggml-cuda/fattn-q2-tile-mixed.cu`
  - `flash_attn_q2_tile_kernel<128>`: pure q2/q2 prefill, Q=8/C=64/K_SUB=8
  - `flash_attn_q2_tile_mixed_mm_kernel<128>`: LP q2 + HP f16 shared online softmax
  - shared K/V row pack per subchunk (`k_pack`/`v_pack`) reused across the Q tile
- env gates:
  - `LLAMA_KV_Q2_TILE_MAIN=1` → route D=128 q2/q2 prefill through tiled kernel
  - `LLAMA_KV_Q2_TILE_MIXED=1` → route mixed q2+f16 fused path through tiled kernel
- scalar fallback remains in `fattn-q2_0-f16.cu` (`for (int j = kv_begin; j < kv_end; ++j)`)
- A/B helper: `scripts/run_q2_tile_ab.sh` (`RUN_REAL=1`)

First RTX 5050 numbers (oscar q2/q2, pp tok/s):

| prompt | vec | tile |
|---:|---:|---:|
| 512 | ~1012 | ~1010 |
| 8192 | ~331 | ~331 |

The skeleton is correctness-first and matches vec speed on this GPU; the next
increment must add FutureMLS-style matmul (dequant once + simd/wmma KQ/V) rather
than more vec-inner-loop patching.

## Expected CUDA Files

Likely files to touch:

- `ggml/src/ggml-cuda/fattn-q2_0-f16.cu`
  - either replace the current scalar per-query body or add a new tiled q2/q2
    implementation next to it
- `ggml/src/ggml-cuda/fattn.cu`
  - route D=128 q2_0/q2_0 prefill to the new kernel behind an env flag
- `ggml/src/ggml-cuda/fattn.cuh`
  - declare the new support and launch functions
- `ggml/src/ggml-cuda/CMakeLists.txt`
  - ensure the new CUDA source is compiled
- optional: `ggml/src/ggml-cuda/fattn-common.cuh`
  - reuse q2 exact decode helpers, but do not make further vec-path micro-tweaks

## Safety

No direct 32k q2 run after CUDA edits. Use:

1. `python3 scripts/run_q2_ramp_next.py`
2. 512 q2 real run only after explicit `--real --ack-real`
3. then 2k, 4k, 8k, 16k via archived ramp evidence
4. 32k q2 remains held by `ACK_Q2_RAMP_GATE_HOLD=1`

The target is not achieved until OSCAR INT2 has valid 32k llama.cpp speed and
peak-VRAM evidence exceeding OSCAR INT4/BF16 speed while retaining the expected
KV memory reduction.
