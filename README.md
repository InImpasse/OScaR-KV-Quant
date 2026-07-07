# OSCAR-KV-Quant llama.cpp Harness

This branch is the **llama.cpp + GGUF** validation path for OSCAR-style KV-cache
experiments. It uses the pinned `third_party/OSCAR` llama.cpp fork and focuses
on Granite 4.0 1B BF16 on a local RTX 5050 class device.

Here, OSCAR means **Offline Spectral Covariance-Aware Rotation**. This branch is
not the SGLang serving harness; it keeps the runtime surface in llama.cpp:
`llama-bench`, `llama-server`, `llama-completion`, GGUF models, and ggml CUDA KV
cache types.

## Current Status

Validated path: **Granite 4.0 1B BF16 GGUF** with optional baked K/V rotation
GGUF.

- **Delivery path**: `oscar_int4` / `plain_int4` using `q4_0/q4_0` KV cache.
- **Speed/memory**: 32K BF16 vs INT4 archived under
  `runs/oscar_int4_bf16_32k_n64_decode_20260624/`.
- **Accuracy**: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME25
  through `llama-server`, using the same table shape as the SGLang harness.
- **Model scope**: all numbers here are local Granite 4.0 1B results, not
  official OSCAR paper numbers.

## Key Takeaways

- INT4 is the current successful llama.cpp route on this hardware.
- At 32K, `oscar_int4` reduces peak memory by about **1.8 GiB** versus BF16
  while preserving prefill throughput in the archived llama.cpp run.
- Accuracy is reported as BF16 vs OSCAR INT4 vs plain INT4 on the same metric
  family used by the SGLang branch.
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
| `scripts/bench_llamacpp_matrix.sh` | SGLang-style preset matrix for llama.cpp |
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

Current delivery evidence is the 32K decode-heavy llama.cpp run:
`runs/oscar_int4_bf16_32k_n64_decode_20260624/`.

| Variant | KV cache | KV pool MiB | Peak MiB | Prefill tok/s | Decode tok/s | Delta peak vs BF16 |
|---|---|---:|---:|---:|---:|---:|
| BF16 | `bf16/bf16` | 2560 | 6143 | 2319.0 | 63.3 | — |
| OSCAR INT4 | `q4_0/q4_0` | 720 | 4307 | 2344.8 | 59.7 | -1836 MiB |

The measured peak drop is almost identical to the KV pool drop, so the memory
win is attributable to the KV-cache representation rather than unrelated arena
noise. Prefill is effectively at BF16 speed; decode is about 94% of BF16 in this
run.

## Accuracy

Benchmarks: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME25.
This llama.cpp branch reports BF16, OSCAR INT4, and plain INT4 on Granite 4.0
1B through `llama-server`, matching the SGLang branch table shape but not the
same backend or KV precision.

Latest clean llama.cpp INT4 rerun:
`runs/granite_accuracy_int4_verify_20260706T101520Z/`.

| Benchmark | Metric | BF16 | OSCAR INT4 | Δ vs BF16 | Plain INT4 | Δ vs BF16 |
|---|---|---:|---:|---:|---:|---:|
| **GPQA** | Score | 28.79 | 26.26 | **-2.53 pt** | 28.79 | **0.00 pt** |
| **GSM8K** | Accuracy | 61.00 | 60.50 | **-0.50 pt** | 56.50 | **-4.50 pt** |
| **MATH500** | Score | 44.40 | 42.60 | **-1.80 pt** | 41.40 | **-3.00 pt** |
| **HumanEval** | Pass@1 | 40.24 | 51.22 | **+10.98 pt** | 41.46 | **+1.22 pt** |
| **HumanEval** | Pass@2 | 48.17 | 58.54 | **+10.37 pt** | 51.83 | **+3.66 pt** |
| **HumanEval** | Pass@5 | 62.80 | 67.07 | **+4.27 pt** | 65.24 | **+2.44 pt** |
| **AIME25** | Score | 6.67 | 6.67 | **0.00 pt** | 3.33 | **-3.33 pt** |

LiveCodeBench v6 did not complete in this clean rerun because the
`livecodebench/code_generation_lite` dataset was not locally cached and the
dataset fetch stalled before loading problems. Historical LCB v6 data from
`runs/granite_accuracy_full_20260705T103055Z/` was `5.71` Pass@1 for BF16,
OSCAR INT4, and plain INT4, but it is not part of the clean rerun above.

The positive HumanEval deltas should be treated as sampling noise rather than
evidence that INT4 improves the model. The useful conclusion is that llama.cpp
`q4_0/q4_0` KV does not show the severe accuracy collapse seen in earlier INT2
experiments.

### SGLang Branch Comparison

The SGLang branch used a different serving stack and INT2-oriented KV path, so
the numbers below are a reference point for behavior, not an apples-to-apples
implementation comparison with llama.cpp INT4.

| Benchmark | Metric | SGLang BF16 | SGLang OSCAR INT2 | Δ vs BF16 | SGLang Plain INT2 | Δ vs BF16 |
|---|---|---:|---:|---:|---:|---:|
| **GPQA** | Score | 23.74 | 24.24 | **+0.50 pt** | 15.66 | **-8.08 pt** |
| **GSM8K** | Accuracy | 56.00 | 54.50 | **-1.50 pt** | 3.00 | **-53.00 pt** |
| **MATH500** | Score | 7.40 | 7.20 | **-0.20 pt** | 0.20 | **-7.20 pt** |
| **LCB V6** | Pass@1 | 7.87 | 6.92 | **-0.95 pt** | 0.00 | **-7.87 pt** |
| **HumanEval** | Pass@1 | 32.93 | 12.68 | **-20.25 pt** | 0.00 | **-32.93 pt** |
| **HumanEval** | Pass@2 | 33.66 | 19.88 | **-13.78 pt** | 0.00 | **-33.66 pt** |
| **HumanEval** | Pass@5 | 34.76 | 32.93 | **-1.83 pt** | 0.00 | **-34.76 pt** |
| **AIME25** | Score | 0.00 | 0.00 | **0.00 pt** | 0.00 | **0.00 pt** |

The main cross-branch takeaway is that SGLang INT2 and llama.cpp INT4 are
different regimes. INT2 is much more aggressive and showed large drops on
plain INT2 and HumanEval, while llama.cpp INT4 remains close to BF16 on the
current non-LCB rerun.

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
