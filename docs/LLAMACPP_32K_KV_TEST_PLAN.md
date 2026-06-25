# llama.cpp 32k KV Test Plan

This repo should stay on the llama.cpp path. Do not run non-llama.cpp harnesses
from this workspace.

## Safety Defaults

- `scripts/bench_32k_llamacpp_kv.sh` defaults to `DRY_RUN=1`; add
  `DRY_RUN=0` only when intentionally launching a real run.
- After a WSL crash or timeout, start with the generated q2 recovery ramp:
  512, 2k, 4k, 8k, 16k, then 32k only if every smaller step is healthy.
- Run one case at a time with `CASES=...`.
- Keep `GEN_TOKENS=1` while validating prefill, VRAM peak, and KV pool size.
- Use `CASE_TIMEOUT_SEC` for every heavy run.
- For WSL recovery runs, set `MAX_PEAK_MIB` so `measure_vram.sh` terminates the
  child process if sampled GPU memory exceeds the cap. `0` disables the cap.
- For repeated single-case runs, set `POST_CASE_COOLDOWN_SEC` so the harness
  waits for GPU memory/utilization to return below the baseline guards before
  continuing. `0` disables the cooldown.
- Real runs refuse to start when baseline GPU memory or utilization is above the
  guards (`MAX_BASELINE_MIB=1024`, `MAX_GPU_UTIL=10` by default). Do not bypass
  this after a q2 timeout unless you have intentionally reset/verified the GPU.
- `scripts/bench_32k_llamacpp_kv.sh` refuses q2/int2 32k cases unless
  `ACK_HEAVY_32K=1` is set.
- Because 32k OSCAR INT2 timed out with empty JSON, 32k q2/int2 now also requires
  `ACK_Q2_32K_NOGO=1`. Prefer code/profiler work before using this override.
- Because the current archived q2 ramp gate is `hold_32k_q2`, real 32k q2/int2
  also requires `ACK_Q2_RAMP_GATE_HOLD=1`. Use this only for deliberate
  post-change validation.
- 32k q2/int2 is single-case only: `CASES=plain_int2` or `CASES=oscar_int2`,
  `GEN_TOKENS=1`, and `REPETITIONS=1`. The harness refuses `CASES=all` or any
  comma-separated case list containing q2 at 32k.
- The timeout path has a low-load smoke check with `measure_vram.sh` wrapping
  `sleep`; it now leaves a summary and cleans up the child process.
- The harness writes `summary.csv` and `summary.md` automatically after real runs.
  Any measured case failure is propagated as the script exit status.

## Ramp

Before generating or running commands, inspect the current readiness summary:

```bash
python3 scripts/report_recovery_readiness.py
python3 scripts/report_q2_ramp_gate.py
```

Prefer generating the ramp commands so the safety flags stay in sync:

```bash
python3 scripts/print_32k_matrix_commands.py
python3 scripts/print_32k_q2_ramp_commands.py
python3 scripts/print_32k_q2_ramp_commands.py --case oscar_int2 --real --ack-real
```

These helpers only print commands; they never execute benchmarks. Their default
output uses `DRY_RUN=1`; printing `DRY_RUN=0` commands requires both `--real`
and `--ack-real`. Both command helpers keep 32k q2/int2 as `DRY_RUN=1` even with
those flags unless both `--ack-32k-q2-real` and `--ack-q2-ramp-gate-hold` are
also supplied. The matrix helper covers the full llama.cpp 32k matrix and keeps
q2/int2 as separate single-case commands. `plain_int3` maps to the llama.cpp
TurboQuant 3-bit KV cache type `turbo3/turbo3`.

The q2 ramp helper is intentionally more cautious than the final 32k matrix:
after a crash it prints 512, 2k, 4k, 8k, 16k, and finally 32k single-case
commands. It also carries `MAX_PEAK_MIB`, `POST_CASE_COOLDOWN_SEC`,
`GEN_TOKENS=1`, and `REPETITIONS=1` through every step.

The q2 ramp gate is also read-only. With the current archived evidence it should
report `recommendation=hold_32k_q2`, because 32k q2 has a known failed row and
valid 32k speed is still missing. Treat that as a stop sign for repeated 32k q2
runs unless there is a code/profiler change to test.

CUDA optimization direction is documented in
`docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md`. The next performance work should port the
FutureMLS Metal tiled mixed-FA idea (`kernel_flash_attn_mixed_mm_q2_0_f16_d128`,
Q=8/C=64 tile reuse) into a CUDA D=128 q2/q2 prefill kernel. Do not spend more
time on current q2 vec-path micro-tweaks or the local scalar per-query
`fattn-q2_0-f16.cu` body as the final performance path.

```bash
# 1. BF16 32k baseline only. This finished once at ~31.9s.
OUT_DIR=/tmp/llamacpp_32k_bf16 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 2. q2/int2 post-crash smoke at 512 first.
OUT_DIR=/tmp/llamacpp_512_plain_int2 \
  PROMPT_TOKENS=512 CASES=plain_int2 CASE_TIMEOUT_SEC=45 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 3. q2/int2 low-load probe at 2k.
OUT_DIR=/tmp/llamacpp_2k_plain_int2 \
  PROMPT_TOKENS=2048 CASES=plain_int2 CASE_TIMEOUT_SEC=60 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 4. q2/int2 low-load probe at 4k.
OUT_DIR=/tmp/llamacpp_4k_plain_int2 \
  PROMPT_TOKENS=4096 CASES=plain_int2 CASE_TIMEOUT_SEC=75 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 5. q2/int2 sanity at 8k.
OUT_DIR=/tmp/llamacpp_8k_plain_int2 \
  PROMPT_TOKENS=8192 CASES=plain_int2 CASE_TIMEOUT_SEC=90 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 6. If 8k is healthy, try 16k single-case.
OUT_DIR=/tmp/llamacpp_16k_plain_int2 \
  PROMPT_TOKENS=16384 CASES=plain_int2 CASE_TIMEOUT_SEC=240 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 \
  scripts/bench_32k_llamacpp_kv.sh

# 7. Only after 16k completes successfully, try one 32k q2/int2 case.
OUT_DIR=/tmp/llamacpp_32k_plain_int2 \
  PROMPT_TOKENS=32768 CASES=plain_int2 CASE_TIMEOUT_SEC=480 \
  MAX_PEAK_MIB=7000 POST_CASE_COOLDOWN_SEC=30 \
  DRY_RUN=0 ACK_HEAVY_32K=1 ACK_Q2_32K_NOGO=1 ACK_Q2_RAMP_GATE_HOLD=1 \
  scripts/bench_32k_llamacpp_kv.sh
```

Summarize a run:

```bash
python3 scripts/summarize_32k_llamacpp_kv.py /tmp/llamacpp_32k_bf16
```

## Cases

- `baseline_bf16`: base GGUF, `bf16/bf16` KV.
- `plain_int2`: base GGUF, `q2_0/q2_0` KV.
- `oscar_int2`: rotated GGUF, `q2_0/q2_0` KV,
  `LLAMA_KV_Q2_0_OWHT=1`, `LLAMA_KV_NO_HADAMARD=1`, K/V clip `0.96/0.92`.
- `oscar_int4`: rotated GGUF, `q4_0/q4_0` KV.
- `plain_int4`: base GGUF, `q4_0/q4_0` KV.
- `plain_int3`: base GGUF, `turbo3/turbo3` KV. This is the llama.cpp
  TurboQuant 3-bit KV cache path; `Q3_K` remains a weight quantization format,
  not a KV cache type.

This is enforced by `scripts/check_kv_cache_types.py`, which checks
`third_party/OSCAR/common/arg.cpp`: `bf16`, `q4_0`, `q2_0`, `turbo2`, and
`turbo3` must be exposed, while 3-bit weight-only types must not be exposed as
KV cache types.

## CUDA Graph Controls

The current `build-cuda` CMake cache has `GGML_CUDA_GRAPHS=ON`, so CUDA graph
support is already compiled into the llama.cpp CUDA backend.

The harness exposes explicit controls for small A/B runs:

- `CUDA_GRAPHS_MODE=auto`: default runtime behavior.
- `CUDA_GRAPHS_MODE=on`: clears `GGML_CUDA_DISABLE_GRAPHS`.
- `CUDA_GRAPHS_MODE=off`: sets `GGML_CUDA_DISABLE_GRAPHS=1`.
- `CUDA_GRAPH_OPT=1`: sets `GGML_CUDA_GRAPH_OPT=1`; default is `0`.

Use these only on low-risk prompts first, for example 512 or 8k single-case
checks. Existing 32k matrix numbers were collected with default graph behavior.
Do not use graph controls as a reason to re-run 32k q2 without a code change;
the current evidence points to q2 attention compute cost, not launch overhead.

Low-risk dry-run helper:

```bash
scripts/cuda_graph_ab.sh
```

Low-risk real A/B, after confirming the GPU is idle:

```bash
RUN_REAL=1 PROMPT_TOKENS=512 CASES=plain_int2 CASE_TIMEOUT_SEC=60 \
  scripts/cuda_graph_ab.sh
```

The helper refuses 32k and multi-case runs. Use it only to decide whether graph
controls are worth testing at 8k later.

One low-risk 512-token real A/B completed after WSL recovered:

```bash
RUN_REAL=1 PROMPT_TOKENS=512 CASES=plain_int2 CASE_TIMEOUT_SEC=60 \
  VRAM_POLL_INTERVAL=0.5 scripts/cuda_graph_ab.sh
```

| graph mode | opt | prompt | KV | peak MiB | pp tok/s | pp vs off | tg tok/s | tg vs off | note |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| off | 0 | 512 | q2_0/q2_0 | 3571 | 2039.0 | 0.00% | 57.2 | 0.00% | baseline for graph A/B |
| on | 1 | 512 | q2_0/q2_0 | 3571 | 2020.6 | -0.90% | 55.9 | -2.23% | no speedup at 512 |

Machine-readable summary: `runs/cuda_graph_ab_512_current/`.

Raw output: `runs/cuda_graph_ab_20260612T062854Z/` (ignored by git). The helper
also writes `graph_ab.csv` and `graph_ab.md` when `RUN_REAL=1`. The GPU returned
to idle immediately after the run.

## Current Evidence

Combined machine-readable report:

- `runs/llamacpp_32k_kv_matrix_current/combined.csv`
- `runs/llamacpp_32k_kv_matrix_current/combined.md`
- archived raw run directories: `runs/llamacpp_32k_kv_matrix_current/raw/`
- current goal audit: `runs/goal_status_current/`
- q2 optimization follow-up: `docs/Q2_32K_OPTIMIZATION_NOTES.md`

Validate the current conclusions without running GPU work:

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

The verifier defaults to no GPU access, including no `nvidia-smi` snapshot. Use
`CHECK_GPU_SNAPSHOT=1 scripts/verify_llamacpp_32k_kv_no_gpu.sh` only when you
want an explicit read-only idle snapshot.

The verifier also runs `scripts/check_llamacpp_only.py` to keep this workspace
on the llama.cpp-only path and reject known non-llama.cpp harness markers across
the top-level workspace and the OSCAR submodule, excluding generated build/raw
trees.
It also runs `scripts/check_build_defaults.py` to keep local scripts pointed at
`third_party/OSCAR/build-cuda`.

Current combined view:

| variant | status | prompt | KV | KV MiB | peak MiB | pp tok/s | tg tok/s | peak saved vs BF16 | KV saved vs BF16 | note |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| oscar_int2 | ok | 16384 | q2_0/q2_0 | 240.0 | 3796 | 183.7 | 28.0 |  |  |  |
| plain_int2 | ok | 16384 | q2_0/q2_0 | 240.0 | 3792 | 180.0 | 44.1 |  |  |  |
| baseline_bf16 | ok | 32768 | bf16/bf16 | 2560.0 | 6160 | 2486.4 | 41.6 | 0.0 | 0.0 |  |
| oscar_int4 | ok | 32768 | q4_0/q4_0 | 720.0 | 4324 | 2533.8 | 39.2 | 1836.0 | 1840.0 |  |
| plain_int4 | ok | 32768 | q4_0/q4_0 | 720.0 | 4324 | 2265.0 | 41.0 | 1836.0 | 1840.0 |  |
| oscar_int2 | failed | 32768 | q2_0/q2_0 | 480.0 | 4036 |  |  | 2124.0 | 2080.0 | missing or invalid llama-bench JSON |

A current 32k BF16 baseline completed with the llama.cpp-only harness:

```bash
OUT_DIR=/tmp/llamacpp_32k_bf16_current \
  PROMPT_TOKENS=32768 CASES=baseline_bf16 GEN_TOKENS=1 CASE_TIMEOUT_SEC=90 \
  DRY_RUN=0 RUN_PREFLIGHT=0 VRAM_POLL_INTERVAL=0.5 \
  scripts/bench_32k_llamacpp_kv.sh
```

| variant | KV | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | duration |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_bf16, 32768 prompt | bf16/bf16 | 2560.0 | 6160 | 6041 | 2486.4 | 41.6 | 31.1s |

The INT4 paths are healthy at 32k. OSCAR INT4 was also checked at 16k as a
lower-risk ramp:

| variant | KV | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | duration |
|---|---|---:|---:|---:|---:|---:|---:|
| oscar_int4, 16384 prompt | q4_0/q4_0 | 360.0 | 3932 | 3813 | 3440.8 | 31.2 | 14.0s |
| oscar_int4, 32768 prompt | q4_0/q4_0 | 720.0 | 4324 | 4205 | 2533.8 | 39.2 | 30.9s |
| plain_int4, 32768 prompt | q4_0/q4_0 | 720.0 | 4324 | 4205 | 2265.0 | 41.0 | 33.0s |

At 32k, BF16 -> INT4 reduces theoretical KV pool by 1840 MiB and observed peak
VRAM by 1836 MiB, so the measured memory drop is now close to the KV cache
storage drop. OSCAR INT4 is slightly faster than plain INT4 on prefill in this
run (2533.8 vs 2265.0 tok/s).

One interrupted 32k smoke produced a valid 2-byte KV baseline. It was collected
before the harness switched `baseline_bf16` from `f16/f16` to `bf16/bf16`, so use
it only as a safety reference, not as the final BF16 row:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|---:|---:|
| baseline_bf16 | 2560.0 | 6106 | 5982 | 2283.1 | 27.7 |

The same smoke showed `plain_int2` did not finish within several minutes before
manual interruption, so q2/int2 work should proceed via the ramp above.

A low-strength 512-token q2 harness check also completed and cleaned up:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|---:|---:|
| plain_int2, 512 prompt, after rebuild | 7.5 | 3570 | 3452 | 1766.6 | 63.9 |

The 512-token check above used the rebuilt `llama-bench` from
`2026-06-12 11:46:33 +0800`, after the latest CUDA source edit.

An 8k single-case q2 ramp check also completed within the 90s timeout:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s |
|---|---:|---:|---:|---:|---:|
| plain_int2, 8192 prompt, after rebuild | 120.0 | 3668 | 3550 | 355.9 | 39.3 |

A 16k single-case q2 ramp did **not** produce valid speed output before the
150s timeout path. The run left an empty `llama-bench` JSON, so only VRAM
sampling is usable:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | status |
|---|---:|---:|---:|---:|---:|---|
| plain_int2, 16384 prompt, after rebuild | 240.0 | 3788 | 3669 |  |  | timeout / invalid JSON |

The previous 16k q2 run held steady at 3788 MiB through 157.6s and produced no
JSON, suggesting the prefill had not completed before timeout rather than a CUDA
crash. Re-running the same single case with a 240s timeout completed:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | duration |
|---|---:|---:|---:|---:|---:|---:|
| plain_int2, 16384 prompt, 240s timeout | 240.0 | 3792 | 3673 | 180.0 | 44.1 | 184.6s |
| oscar_int2, 16384 prompt, 240s timeout | 240.0 | 3796 | 3670 | 183.7 | 28.0 | 183.7s |

The 16k q2 gate now has valid JSON for both plain and OSCAR INT2, but q2 is
still extremely slow. OSCAR rotation does not materially change prefill speed at
16k, so the bottleneck remains the q2 CUDA attention path rather than the
rotation metadata. Any 32k q2/int2 attempt must remain single-case,
`GEN_TOKENS=1`, GPU idle, and guarded by `ACK_HEAVY_32K=1`,
`ACK_Q2_32K_NOGO=1`, `ACK_Q2_RAMP_GATE_HOLD=1`, plus a long timeout. Do not run
`all` with q2 enabled.

A single 32k OSCAR INT2 attempt with a 480s timeout did not finish:

| variant | KV pool MiB | peak MiB | delta MiB | pp tok/s | tg tok/s | duration | status |
|---|---:|---:|---:|---:|---:|---:|---|
| oscar_int2, 32768 prompt, 480s timeout | 480.0 | 4036 | 3910 |  |  | 508.7s | timeout / invalid JSON |

The run left an empty `llama-bench` JSON and returned no speed output. After the
timeout, no compute app remained and VRAM returned to ~126 MiB, but
`nvidia-smi` still reported 100% GPU utilization. Treat the WSL GPU state as
suspect after this timeout; do not run more GPU benchmarks until utilization
returns to idle or WSL/GPU state is reset. The harness now checks baseline GPU
utilization before real runs and requires `ACK_Q2_32K_NOGO=1` plus
`ACK_Q2_RAMP_GATE_HOLD=1` for any further 32k q2/int2 attempt, so this failure
mode is caught before launching another benchmark. This 32k q2 result is a
NO-GO for the current exact q2 path under the safety limits used here.
