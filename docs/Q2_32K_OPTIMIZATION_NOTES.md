# Q2 32k Optimization Notes

This note captures the current llama.cpp-only q2 state after the 32k harness
ramp. It is meant to prevent repeating unsafe long runs or already-failed CUDA
experiments.

## Current Evidence

Source of truth:

- `runs/llamacpp_32k_kv_matrix_current/combined.csv`
- `runs/llamacpp_32k_kv_matrix_current/combined.md`
- archived raw run directories: `runs/llamacpp_32k_kv_matrix_current/raw/`
- `docs/LLAMACPP_32K_KV_TEST_PLAN.md`
- `docs/EXTERNAL_REFERENCE_COMPARISON.md`

No-GPU consistency check:

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

The verifier includes `scripts/check_q2_cuda_static.py`, which checks that the
current q2 CUDA path is still the restored exact LUT baseline, that q2 KQ keeps
the expected 3-dp4a exact reconstruction while q4 KQ remains a 1-dp4a path, and
that known failed experiment flags have not been reintroduced.

It also includes `scripts/audit_goal_status.py`, which intentionally reports the
overall goal as incomplete until 32k INT2 has a valid speed result. Archived
status report: `runs/goal_status_current/`.

Key results:

| variant | prompt | status | KV | KV MiB | peak MiB | pp tok/s | note |
|---|---:|---|---|---:|---:|---:|---|
| baseline_bf16 | 32768 | ok | bf16/bf16 | 2560.0 | 6160 | 2486.4 | final BF16 baseline |
| oscar_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2533.8 | memory drop matches KV drop |
| plain_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2265.0 | healthy |
| plain_int2 | 16384 | ok | q2_0/q2_0 | 240.0 | 3792 | 180.0 | valid 16k gate |
| oscar_int2 | 16384 | ok | q2_0/q2_0 | 240.0 | 3796 | 183.7 | rotation does not change speed materially |
| oscar_int2 | 32768 | failed | q2_0/q2_0 | 480.0 | 4036 |  | 480s timeout, empty JSON |

The 32k OSCAR INT2 attempt is a NO-GO for the current exact q2 path under the
safety limits used here. Further 32k q2 runs require both `ACK_HEAVY_32K=1` and
`ACK_Q2_32K_NOGO=1`.

## Current q2 CUDA Path

The hot q2 flash-attention path is in:

- `third_party/OSCAR/ggml/src/ggml-cuda/fattn-common.cuh`
- `third_party/OSCAR/ggml/src/ggml-cuda/fattn-vec.cuh`
- `third_party/OSCAR/ggml/src/ggml-cuda/fattn.cu`

Current KQ q2 implementation:

- `vec_dot_fattn_vec_KQ_q2_0`
- packed q2 byte lookup via `Q2_0_FATTN_SIGN_LUT` and `Q2_0_FATTN_HIGH_LUT`
- `sum_sign = dp4a(sign, q)`
- `sum_high = dp4a(high, q)`
- `usum = dp4a(ones, q)`
- final exact reconstruction with block scale `d` and replicated/group mean `m`

This is exact, but it is structurally more expensive than q4 KQ. q4 needs one
`dp4a`; exact q2 needs multiple integer dot pieces plus mean handling.

V q2 still dequantizes through `dequantize_V_q2_0` with scalar q2 decode per
lane. Prior `ne == 4` V fma rewriting regressed badly and must not be repeated
without profiler evidence.

## What The Results Imply

- INT4 is already good: at 32k, BF16 -> INT4 reduces theoretical KV by 1840 MiB
  and observed peak VRAM by 1836 MiB.
- q2 memory is smaller, but prefill is unusably slow at long context.
- OSCAR rotation is not the speed issue: 16k plain INT2 and OSCAR INT2 are
  effectively tied on prefill.
- CUDA graph is unlikely to rescue 32k q2 prefill by itself. The failure is a
  long, compute-heavy q2 attention path, not repeated launch overhead. It may
  help decode or many small launches, but 32k prefill needs kernel-level q2 work.
- `build-cuda` already has `GGML_CUDA_GRAPHS=ON`. The harness can now force
  graph mode for low-risk A/B runs with `CUDA_GRAPHS_MODE=on` and
  `CUDA_GRAPH_OPT=1`, but this should be validated at 512/8k before any heavier
  run.
- Use `scripts/cuda_graph_ab.sh` for graph dry-runs or 512-token real A/B. It
  refuses 32k by design.
- A 512-token `plain_int2` graph A/B showed no benefit: graph off was
  2039.0 pp tok/s, graph on with `GGML_CUDA_GRAPH_OPT=1` was 2020.6 pp tok/s
  (-0.90%). This does not rule out decode benefits, but it is not evidence for
  re-running 32k q2. Machine-readable summary:
  `runs/cuda_graph_ab_512_current/`.

## Do Not Repeat

These experiments were already tried or ruled out by data:

- `GGML_CUDA_Q2_FATTN_FAST` single-dp4a approximation: no useful speedup.
- Shared-K / `GGML_CUDA_Q2_FATTN_TILE_D128`: no useful speedup.
- `ncols_partial`: no useful speedup.
- D=128 KQ inline with `dm[4][2]` register cache: large regression.
- V q2 `ne == 4` fma rewrite: large regression.
- Re-running 32k q2 without code changes: expected to waste minutes and may
  leave WSL/NVML utilization in a suspect state after timeout.

## Next Viable Work

Preferred order:

1. Fix profiler access, then collect `ncu` for
   `flash_attn_ext_vec<128,4,q2_0,q2_0>` on 8k or 16k first.
2. If profiler remains unavailable, keep tests at 8k/16k and use very narrow
   compile-time experiments with a hard gate: 8k q2 must not regress below the
   restored LUT baseline and should improve by at least 10%.
3. Consider a separate q2/q2 D=128 kernel only if profiler indicates a plausible
   2x+ path. Keep it behind a default-off env/compile flag.
4. For product-facing 32k today, prefer INT4. It already matches the KV memory
   savings target and keeps prefill comparable to BF16.

Profiling helpers default to dry-run and the current CUDA build:

```bash
DRY_RUN=1 scripts/q2_profile.sh
DRY_RUN=1 scripts/q2_segment_bench.sh
```

Set `DRY_RUN=0` only after confirming the GPU is idle and the WSL profiler
preflight is fixed.

`scripts/check_q2_profile_safety.py` verifies these defaults and ensures
`q2_profile.sh` exits before profiler preflight when `DRY_RUN=1`.

Short version: exact q2_0 is currently a memory win but a 32k prefill speed
NO-GO. Beating the external reference result will require a real q2 CUDA kernel
change or a different/approximate KV format, not another harness run.
