# OSCAR-KV-Quant llama.cpp Harness

This branch is the **llama.cpp + GGUF** validation path for OSCAR-style KV-cache
experiments. It uses the pinned `third_party/OSCAR` llama.cpp fork and focuses
on Granite 4.0 1B BF16 on a local RTX 5050 class device.

Here, OSCAR means **Offline Spectral Covariance-Aware Rotation**. This branch
keeps the runtime surface in llama.cpp: `llama-bench`, `llama-server`,
`llama-completion`, GGUF models, and ggml CUDA KV cache types.

## Current Status

Validated path: **Granite 4.0 1B BF16 GGUF** with optional baked K/V rotation
GGUF.

- **Delivery path**: `oscar_int4` / `plain_int4` using `q4_0/q4_0` KV cache.
- **Speed/memory**: 512 / 2K / 8K / 16K / 32K BF16 vs INT4 sweep archived
  under `runs/llamacpp_int4_context_sweep_20260707T084527Z/`.
- **Accuracy**: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME25
  through `llama-server`.
- **Model scope**: all numbers here are local Granite 4.0 1B results, not
  official OSCAR paper numbers.

## Key Takeaways

- INT4 is the current recommended llama.cpp route on this hardware.
- At 32K, `oscar_int4` reduces peak memory by about **1.8 GiB** versus BF16
  while preserving prefill throughput in the archived llama.cpp run.
- Accuracy is reported as BF16 vs OSCAR INT4 vs plain INT4, with a reference
  comparison to the SGLang INT2 branch.
- Plain INT4 is useful as a quantized-control baseline, but the delivery path is
  OSCAR INT4 with the rotated GGUF.

## Requirements

- Linux or WSL2 with NVIDIA GPU access
- CUDA-capable llama.cpp build; current local validation used CUDA 12.9
- `nvidia-smi` for VRAM sampling
- Python 3.10+ for accuracy scripts
- Granite GGUF files:
  - `checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`
  - `checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf`

Optional accuracy dependencies:

```bash
python3 -m pip install datasets pandas requests tqdm
python3 -m pip install git+https://github.com/openai/human-eval.git
```

For LiveCodeBench v6:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git third_party/LiveCodeBench
python3 -m pip install -e third_party/LiveCodeBench
```

## Quick Start

Initialize and build the llama.cpp fork:

```bash
git submodule update --init --recursive

cmake -S third_party/OSCAR -B third_party/OSCAR/build-cuda \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON

cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench llama-server
```

Check the GPU-visible build:

```bash
third_party/OSCAR/build-cuda/bin/llama-bench --list-devices
```

Run a dry-run matrix first:

```bash
PRESETS=short MODES=bf16,oscar-int4,int4 \
  scripts/bench_llamacpp_matrix.sh
```

Run a real short/medium/long matrix:

```bash
OUT_ROOT=runs/llamacpp_bench_matrix_$(date +%Y%m%d_%H%M%S) \
PRESETS=short,medium,long \
MODES=bf16,oscar-int4,int4 \
DRY_RUN=0 \
CASE_TIMEOUT_SEC=300 \
  scripts/bench_llamacpp_matrix.sh
```

## Repository Layout

| Path | Purpose |
|---|---|
| `third_party/OSCAR/` | llama.cpp fork with OSCAR INT4 delivery path |
| `scripts/bench_32k_llamacpp_kv.sh` | low-level guarded llama-bench runner |
| `scripts/bench_llamacpp_matrix.sh` | preset matrix for llama.cpp |
| `scripts/cuda_graph_compare_llamacpp_matrix.sh` | graph off/on preset matrix |
| `scripts/run_llamacpp_latency_matrix.sh` | llama-server streaming first-token/P95 latency matrix |
| `scripts/run_llamacpp_accuracy_suite.sh` | GPQA/GSM8K/MATH500/HumanEval/AIME25 through llama-server |
| `scripts/run_llamacpp_lcb_v6.sh` | LiveCodeBench v6 through llama-server |
| `runs/*_current/` | selected archived validation outputs |
| `docs/` | design notes, triage notes, and delivery summaries |

## KV Modes

| Variant | Model | KV cache | Notes |
|---|---|---|---|
| `baseline_bf16` | BF16 GGUF | `bf16/bf16` | accuracy and memory baseline |
| `plain_int4` | BF16 GGUF | `q4_0/q4_0` | quantized-control baseline |
| `oscar_int4` | rotated GGUF | `q4_0/q4_0` | current delivery-oriented route |

These modes quantize **KV cache storage**, not model weights. The baseline model
weights remain BF16.

## Speed And Memory

Current context-sweep evidence:
`runs/llamacpp_int4_context_sweep_20260707T084527Z/`.

The sweep uses `GEN_TOKENS=1` and is intended for KV pool, peak memory, and
prefill throughput. Decode-heavy behavior is still represented by the archived
32K n64 run `runs/oscar_int4_bf16_32k_n64_decode_20260624/`.

### KV Pool

| Context | BF16 | OSCAR INT4 | Delta vs BF16 | plain INT4 | Delta vs BF16 |
|---|---:|---:|---:|---:|---:|
| 512 | 40 MiB | 11.25 MiB | -72% | 11.25 MiB | -72% |
| 2K | 160 MiB | 45 MiB | -72% | 45 MiB | -72% |
| 8K | 640 MiB | 180 MiB | -72% | 180 MiB | -72% |
| 16K | 1280 MiB | 360 MiB | -72% | 360 MiB | -72% |
| 32K | 2560 MiB | 720 MiB | -72% | 720 MiB | -72% |

### Peak Memory

| Context | BF16 | OSCAR INT4 | Delta vs BF16 | plain INT4 | Delta vs BF16 |
|---|---:|---:|---:|---:|---:|
| 512 | 3424 MiB | 3400 MiB | -1% / -24 MiB | 3610 MiB | +5% / +186 MiB |
| 2K | 3810 MiB | 3690 MiB | -3% / -120 MiB | 3696 MiB | -3% / -114 MiB |
| 8K | 4286 MiB | 3830 MiB | -11% / -456 MiB | 3830 MiB | -11% / -456 MiB |
| 16K | 4970 MiB | 4026 MiB | -19% / -944 MiB | 4026 MiB | -19% / -944 MiB |
| 32K | 6254 MiB | 4418 MiB | -29% / -1836 MiB | 4418 MiB | -29% / -1836 MiB |

### Prefill

| Context | BF16 | OSCAR INT4 | Delta vs BF16 | plain INT4 | Delta vs BF16 |
|---|---:|---:|---:|---:|---:|
| 512 | 12845 tok/s | 11464 tok/s | -11% | 12327 tok/s | -4% |
| 2K | 15292 tok/s | 13758 tok/s | -10% | 14652 tok/s | -4% |
| 8K | 12420 tok/s | 11428 tok/s | -8% | 12072 tok/s | -3% |
| 16K | 9806 tok/s | 9264 tok/s | -6% | 9675 tok/s | -1% |
| 32K | 6830 tok/s | 6632 tok/s | -3% | 6821 tok/s | 0% |

The INT4 KV pool is consistently about 72% smaller than BF16. The peak-memory
benefit becomes more visible at longer context; at 32K, OSCAR INT4 saves
1836 MiB peak memory in this sweep while staying within about 3% of BF16
prefill throughput.

## Accuracy

Benchmarks: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME25.
The table below compares the current **llama.cpp branch** (`BF16`, `OSCAR
INT4`, `plain INT4`) against the **SGLang branch** (`BF16`, `OSCAR INT2`,
`plain INT2`). Cross-branch absolute scores are not comparable; only
within-branch deltas are meaningful.

Latest clean llama.cpp INT4 rerun:
`runs/granite_accuracy_int4_verify_20260706T101520Z/`.

Use the delta columns to compare each quantized mode with the BF16 baseline
from the same branch. The llama.cpp and SGLang BF16 columns use different
serving, prompting, and grading paths, so their absolute scores should not be
used to draw KV-cache conclusions.

| Benchmark | Metric | llama.cpp BF16 | llama.cpp OSCAR INT4 (Δ) | llama.cpp Plain INT4 (Δ) | SGLang BF16 | SGLang OSCAR INT2 (Δ) | SGLang Plain INT2 (Δ) |
|---|---|---:|---:|---:|---:|---:|---:|
| **GPQA** | Score | 28.79 | 26.26 (-2.53) | 28.79 (+0.00) | 23.74 | 24.24 (+0.50) | 15.66 (-8.08) |
| **GSM8K** | Accuracy | 61.00 | 60.50 (-0.50) | 56.50 (-4.50) | 56.00 | 54.50 (-1.50) | 3.00 (-53.00) |
| **MATH500** | Score | 44.40 | 42.60 (-1.80) | 41.40 (-3.00) | 7.40 | 7.20 (-0.20) | 0.20 (-7.20) |
| **LCB V6** | Pass@1 | 5.71* | 5.71* (+0.00) | 5.71* (+0.00) | 7.87 | 6.92 (-0.95) | 0.00 (-7.87) |
| **HumanEval** | Pass@1 | 40.24 | 51.22 (+10.98) | 41.46 (+1.22) | 32.93 | 12.68 (-20.25) | 0.00 (-32.93) |
| **HumanEval** | Pass@2 | 48.17 | 58.54 (+10.37) | 51.83 (+3.66) | 33.66 | 19.88 (-13.78) | 0.00 (-33.66) |
| **HumanEval** | Pass@5 | 62.80 | 67.07 (+4.27) | 65.24 (+2.44) | 34.76 | 32.93 (-1.83) | 0.00 (-34.76) |
| **AIME25** | Score | 6.67 | 6.67 (+0.00) | 3.33 (-3.33) | 0.00 | 0.00 (+0.00) | 0.00 (+0.00) |

`*` LiveCodeBench v6 was not included in the clean llama.cpp rerun because the
`livecodebench/code_generation_lite` dataset was not locally cached and the
dataset fetch did not complete. The llama.cpp LCB values shown above are from
the historical run `runs/granite_accuracy_full_20260705T103055Z/`.

The main cross-branch takeaway is that SGLang INT2 and llama.cpp INT4 represent
different regimes. INT2 is substantially more aggressive and shows larger
regressions for plain INT2 and HumanEval, while llama.cpp `q4_0/q4_0` KV remains
close to BF16 on the current non-LCB rerun. The large BF16 gaps between
branches, especially on MATH500, point to harness, prompt, and grading
differences rather than KV-cache effects. The positive HumanEval deltas should
be interpreted as sampling variance, not as evidence that INT4 improves model
accuracy.

Plain INT4 is a quantized-control baseline, not a stronger variant. In this
rerun it is higher than OSCAR INT4 only on GPQA (`57/198` vs `52/198`, a
5-question difference); OSCAR INT4 is higher or equal on GSM8K, MATH500,
HumanEval, and AIME25. This GPQA gap is within the noise expected from a
198-question sample, and the OSCAR rotation should not be expected to improve
every metric monotonically.

Accuracy outputs are generated locally under `runs/<accuracy_dir>/`:

| Benchmark | Metric | Runner |
|---|---|---|
| **GPQA** | Score | `llama-eval.py` |
| **GSM8K** | Accuracy | `llama-eval.py` |
| **MATH-500** | Score | `llama-eval.py` |
| **HumanEval** | Pass@1/2/5 | `llama-eval.py` with `human-eval` grader |
| **LCB V6** | Pass@1 | official LiveCodeBench runner through `llama-server` |
| **AIME25** | Score | `llama-eval.py` (`opencompass/AIME2025`) |

## How To Rerun Accuracy

Check the full Granite INT4 plan without running it:

```bash
OUT_DIR=/tmp/granite_accuracy_plan \
DRY_RUN=1 \
  scripts/run_granite_accuracy_full.sh
```

Run the full Granite suite across BF16, OSCAR INT4, and plain INT4:

```bash
OUT_DIR=runs/granite_accuracy_full_$(date +%Y%m%d_%H%M%S) \
DRY_RUN=0 \
ACK_EVAL=1 \
ALLOW_HUMANEVAL_EXEC=1 \
ALLOW_CODE_EXEC=1 \
  scripts/run_granite_accuracy_full.sh
```

Run the full non-LCB suite first, useful when LiveCodeBench needs a separate
environment:

```bash
OUT_DIR=runs/granite_accuracy_int4_non_lcb_full_$(date +%Y%m%d_%H%M%S) \
VARIANTS=baseline_bf16,oscar_int4,plain_int4 \
NON_LCB_DATASETS=gpqa,gsm8k,math500,humaneval,aime2025 \
GPQA_N_CASES=198 GSM8K_N_CASES=200 MATH500_N_CASES=500 AIME25_N_CASES=30 \
HUMANEVAL_N_CASES=164 HUMANEVAL_SAMPLES=5 RUN_LCB=0 \
DRY_RUN=0 ACK_EVAL=1 ALLOW_HUMANEVAL_EXEC=1 \
  scripts/run_granite_accuracy_full.sh
```

Run only LiveCodeBench v6:

```bash
OUT_DIR=runs/llamacpp_lcb_v6_$(date +%Y%m%d_%H%M%S) \
VARIANTS=baseline_bf16,oscar_int4,plain_int4 \
LIVE_CODE_BENCH_ROOT=third_party/LiveCodeBench \
LCB_RELEASE=release_v6 \
LCB_N=1 \
DRY_RUN=0 \
ACK_EVAL=1 \
ALLOW_CODE_EXEC=1 \
  scripts/run_llamacpp_lcb_v6.sh
```

Common knobs:

| Env | Default | Purpose |
|---|---:|---|
| `THREADS` | `nproc` | evaluator request workers; use 16-32 first on one 40GB GPU |
| `SERVER_PARALLEL` | `1` | llama-server parallel slots; try 2-4 before higher values |
| `RUN_LCB` | `1` | set `0` to skip LiveCodeBench v6 |
| `RESUME` / `SKIP_COMPLETED` | `1` / `1` | continue partial JSONs and skip completed variants |
| `HUMANEVAL_SAMPLES` | `5` | repeated HumanEval samples used for Pass@1/2/5 |
| `CHECK_DATASETS` | `1` | set `0` if datasets are already cached or the machine is offline |
| `LIVE_CODE_BENCH_ROOT` | `third_party/LiveCodeBench` | LiveCodeBench checkout path |

## Accuracy Troubleshooting

- Overseas or unrestricted networks should use the official Hugging Face Hub
  directly. Do not set `HF_ENDPOINT` unless you intentionally want a mirror.
- In China or restricted networks, configure a local proxy or persistent
  Hugging Face cache before loading datasets.
- `Idavidrein/gpqa` is gated. Use an authenticated Hugging Face cache/token, or
  the CSV fallback in the vendored `llama-eval.py` path.
- If `lcb_runner` imports fail with `ModuleNotFoundError: anthropic`, install the
  missing optional dependency in the LiveCodeBench Python environment.
- If LiveCodeBench reports `KeyError: 'granite-4.0-1b-base'`, keep
  `LCB_MODEL_NAME=granite-4.0-1b-base`; the wrapper resolves it to an available
  LiveCodeBench adapter while requests still go to local `llama-server`.

## Verification

Run the no-GPU consistency checks:

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

Useful lightweight checks:

```bash
python3 scripts/check_execution_entrypoints.py
bash -n scripts/bench_32k_llamacpp_kv.sh scripts/run_llamacpp_accuracy_suite.sh scripts/run_granite_accuracy_full.sh
python3 -m py_compile scripts/summarize_llamacpp_matrix.py scripts/summarize_llamacpp_accuracy_suite.py scripts/summarize_granite_accuracy_full.py
```

## Limitations

- The current successful llama.cpp long-context delivery result is INT4.
- Full accuracy runs can be slow on a laptop GPU.
- HumanEval and LiveCodeBench execute generated code during grading; the scripts
  require explicit acknowledgements before real runs.
- Rotation files under `rotation/**/rotations/` are treated as local artifacts
  and are ignored by the parent repo.
