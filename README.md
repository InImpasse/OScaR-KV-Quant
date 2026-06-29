# OSCAR-KV-Quant llama.cpp Harness

This branch is the **llama.cpp + GGUF** validation path for OSCAR-style KV-cache
experiments. It uses the pinned `third_party/OSCAR` llama.cpp fork and focuses on
Granite 4.0 1B BF16 on a local RTX 5050 8GB class device.

Here, OSCAR means **Offline Spectral Covariance-Aware Rotation**. This branch is
not the SGLang serving harness; it keeps the runtime surface in llama.cpp:
`llama-bench`, `llama-server`, `llama-completion`, GGUF models, and ggml CUDA KV
cache types.

## Current status

Validated path: **Granite 4.0 1B BF16 GGUF** with optional baked K/V rotation
GGUF.

- **Delivery path**: `oscar_int4` / `plain_int4` using `q4_0/q4_0` KV cache.
- **INT2 status**: exact `q2_0/q2_0` and OSCAR INT2 remain experimental for
  llama.cpp long-prefill speed; 32K q2 is guarded as a known no-go until kernel
  work changes.
- **Speed/memory**: 32K BF16 vs INT4 archived under
  `runs/llamacpp_32k_kv_matrix_current/`.
- **CUDA graph**: 512-token q2 graph A/B is archived under
  `runs/cuda_graph_ab_512_current/`; it did not rescue q2 prefill.
- **Accuracy**: llama.cpp wrappers now cover GPQA, GSM8K, MATH-500, HumanEval,
  AIME25, and LiveCodeBench v6 through `llama-server`.

## Key takeaways

- INT4 is the current successful llama.cpp route on this hardware. At 32K,
  `q4_0/q4_0` reduces peak memory by about **1.8 GiB** versus BF16 while keeping
  prefill essentially at BF16 speed in the archived run.
- Exact q2 / INT2 is a memory win but not a 32K llama.cpp speed win today.
  The archived 32K `oscar_int2` run timed out with empty JSON, and 16K q2
  prefill stayed around 180 tok/s.
- OSCAR rotation by itself is not the q2 speed bottleneck. Plain and rotated q2
  show the same long-context speed class.
- CUDA graph is useful for many serving workloads, but current llama.cpp q2
  long-prefill is dominated by kernel work rather than launch overhead.
- The next q2 speed direction is a dedicated D=128 tiled q2/q2 prefill kernel
  inspired by FutureMLS Metal mixed-FA ideas, not small scalar dequant tweaks.

## Requirements

- Linux or WSL2 with NVIDIA GPU access
- CUDA-capable llama.cpp build; current local validation used CUDA 12.9
- `nvidia-smi` for VRAM sampling
- Python 3.10+ for scripts
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

## Quick start

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

## Repository layout

| Path | Purpose |
|---|---|
| `third_party/OSCAR/` | llama.cpp fork with OSCAR INT4 delivery path and INT2 research history |
| `scripts/bench_32k_llamacpp_kv.sh` | low-level guarded llama-bench runner |
| `scripts/bench_llamacpp_matrix.sh` | SGLang-style preset matrix for llama.cpp |
| `scripts/cuda_graph_compare_llamacpp_matrix.sh` | graph off/on preset matrix |
| `scripts/run_llamacpp_latency_matrix.sh` | llama-server streaming first-token/P95 latency matrix |
| `scripts/run_llamacpp_accuracy_suite.sh` | GPQA/GSM8K/MATH500/HumanEval/AIME25 through llama-server |
| `scripts/run_llamacpp_lcb_v6.sh` | LiveCodeBench v6 through llama-server |
| `runs/*_current/` | selected archived validation outputs |
| `docs/` | design notes, triage notes, and delivery summaries |

## KV modes

| Variant | Model | KV cache | Notes |
|---|---|---|---|
| `baseline_bf16` | BF16 GGUF | `bf16/bf16` | accuracy and memory baseline |
| `plain_int2` | BF16 GGUF | `q2_0/q2_0` | exact q2 control; slow at long prefill |
| `oscar_int2` | rotated GGUF | `q2_0/q2_0` | rotated q2 control; still slow at 32K |
| `plain_int4` | BF16 GGUF | `q4_0/q4_0` | healthy INT4 control |
| `oscar_int4` | rotated GGUF | `q4_0/q4_0` | current delivery-oriented route |

These flags quantize **KV cache storage**, not model weights. The baseline model
weights remain BF16.

## Speed and memory

Current-device llama.cpp INT2 matrix:

```bash
runs/llamacpp_cuda_graph_compare_matrix_current_device_20260626_102435/on/matrix.md
runs/llamacpp_cuda_graph_compare_matrix_current_device_20260626_102435/off/matrix.md
runs/llamacpp_cuda_graph_compare_matrix_current_device_20260626_102435/on/bench_compatible.csv
runs/llamacpp_cuda_graph_compare_matrix_current_device_20260626_102435/off/bench_compatible.csv
```

Measurement notes:

- Hardware: local RTX 5050 8GB class device.
- Variants: `baseline_bf16`, `oscar_int2`, and `plain_int2`.
- `Prefill`, `Steady`, `Peak`, and `KV pool` come from `llama-bench`.
- `Decode first` and `P95` come from `llama-server` streaming with 64 generated
  tokens and are overlaid into the same table shape.
- 32K INT2 cases are guarded and were run one variant at a time.

### CUDA graph on

#### Decode first (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.67 | 1.94 | -27% | 1.67 | -37% |
| 2K | 2.37 | 0.50 | -79% | 0.44 | -81% |
| 8K | 0.52 | 0.04 | -91% | 0.04 | -92% |
| 16K | 0.21 | 0.01 | -94% | 0.01 | -94% |
| 32K | 0.08 | 0.00 | -96% | 0.00 | -96% |

#### Steady (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 57.41 | 58.90 | +3% | 62.78 | +9% |
| 2K | 78.36 | 63.56 | -19% | 61.97 | -21% |
| 8K | 69.31 | 51.06 | -26% | 53.75 | -22% |
| 16K | 62.30 | 43.37 | -30% | 50.69 | -19% |
| 32K | 60.95 | 51.41 | -16% | 58.11 | -5% |

#### Peak (MiB, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 3476 | 3442 | -1% (-34 MiB) | 3448 | -1% (-28 MiB) |
| 2K | 3596 | 3464 | -4% (-132 MiB) | 3470 | -4% (-126 MiB) |
| 8K | 4074 | 3550 | -13% (-524 MiB) | 3550 | -13% (-524 MiB) |
| 16K | 4758 | 3670 | -23% (-1088 MiB) | 3670 | -23% (-1088 MiB) |
| 32K | 6042 | 3910 | -35% (-2132 MiB) | 3910 | -35% (-2132 MiB) |

#### KV pool K+V (MiB, measured/estimated, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 40 | 8 | -81% (-32 MiB) | 8 | -81% (-32 MiB) |
| 2K | 160 | 30 | -81% (-130 MiB) | 30 | -81% (-130 MiB) |
| 8K | 640 | 120 | -81% (-520 MiB) | 120 | -81% (-520 MiB) |
| 16K | 1280 | 240 | -81% (-1040 MiB) | 240 | -81% (-1040 MiB) |
| 32K | 2560 | 480 | -81% (-2080 MiB) | 480 | -81% (-2080 MiB) |

#### Prefill (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 4834 | 2083 | -57% | 2136 | -56% |
| 2K | 5322 | 1067 | -80% | 1092 | -79% |
| 8K | 4299 | 327 | -92% | 330 | -92% |
| 16K | 3579 | 177 | -95% | 180 | -95% |
| 32K | 2543 | 92 | -96% | 93 | -96% |

#### P95 (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 38.78 | 38.59 | -0% | 34.90 | -10% |
| 2K | 43.52 | 34.31 | -21% | 26.92 | -38% |
| 8K | 50.52 | 23.59 | -53% | 24.45 | -52% |
| 16K | 50.36 | 15.38 | -69% | 15.71 | -69% |
| 32K | 40.75 | 8.54 | -79% | 10.00 | -75% |

### CUDA graph off

#### Decode first (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.81 | 2.18 | +169% | 1.67 | +107% |
| 2K | 2.11 | 0.50 | -76% | 0.52 | -75% |
| 8K | 0.51 | 0.05 | -91% | 0.04 | -91% |
| 16K | 0.21 | 0.01 | -94% | 0.01 | -94% |
| 32K | 0.08 | 0.00 | -96% | 0.00 | -96% |

#### Steady (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 56.30 | 55.11 | -2% | 53.20 | -6% |
| 2K | 67.34 | 62.71 | -7% | 60.89 | -10% |
| 8K | 69.14 | 54.29 | -21% | 53.83 | -22% |
| 16K | 62.05 | 51.61 | -17% | 52.48 | -15% |
| 32K | 61.90 | 52.88 | -15% | 51.10 | -17% |

#### Peak (MiB, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 3476 | 3442 | -1% (-34 MiB) | 3448 | -1% (-28 MiB) |
| 2K | 3596 | 3464 | -4% (-132 MiB) | 3470 | -4% (-126 MiB) |
| 8K | 4074 | 3550 | -13% (-524 MiB) | 3550 | -13% (-524 MiB) |
| 16K | 4758 | 3670 | -23% (-1088 MiB) | 3670 | -23% (-1088 MiB) |
| 32K | 6042 | 3910 | -35% (-2132 MiB) | 3910 | -35% (-2132 MiB) |

#### KV pool K+V (MiB, measured/estimated, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 40 | 8 | -81% (-32 MiB) | 8 | -81% (-32 MiB) |
| 2K | 160 | 30 | -81% (-130 MiB) | 30 | -81% (-130 MiB) |
| 8K | 640 | 120 | -81% (-520 MiB) | 120 | -81% (-520 MiB) |
| 16K | 1280 | 240 | -81% (-1040 MiB) | 240 | -81% (-1040 MiB) |
| 32K | 2560 | 480 | -81% (-2080 MiB) | 480 | -81% (-2080 MiB) |

#### Prefill (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 5273 | 2058 | -61% | 2138 | -59% |
| 2K | 5252 | 1095 | -79% | 1113 | -79% |
| 8K | 4430 | 375 | -92% | 327 | -93% |
| 16K | 3650 | 185 | -95% | 185 | -95% |
| 32K | 2022 | 93 | -95% | 93 | -95% |

#### P95 (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Delta vs BF16 | plain INT2 | Delta vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 512 | 38.64 | 38.09 | -1% | 39.14 | +1% |
| 2K | 46.89 | 33.25 | -29% | 35.89 | -23% |
| 8K | 56.77 | 24.47 | -57% | 21.72 | -62% |
| 16K | 49.34 | 14.68 | -70% | 14.29 | -71% |
| 32K | 40.94 | 10.06 | -75% | 10.01 | -76% |

Run the graph comparison:

```bash
OUT_ROOT=runs/llamacpp_cuda_graph_compare_matrix_$(date +%Y%m%d_%H%M%S) \
PRESETS=short,medium,long \
MODES=bf16,oscar-int4,int4 \
RUN_REAL=1 \
CASE_TIMEOUT_SEC=300 \
  scripts/cuda_graph_compare_llamacpp_matrix.sh
```

32K q2/int2 is intentionally guarded:

```bash
OUT_ROOT=runs/llamacpp_cuda_graph_compare_matrix_32k_$(date +%Y%m%d_%H%M%S) \
PRESETS=16k,32k \
MODES=bf16,oscar-int2,int2 \
RUN_REAL=1 \
ACK_HEAVY_32K=1 \
ACK_Q2_32K_NOGO=1 \
ACK_Q2_RAMP_GATE_HOLD=1 \
CASE_TIMEOUT_SEC=900 \
  scripts/cuda_graph_compare_llamacpp_matrix.sh
```

Summarize:

```bash
python3 scripts/summarize_llamacpp_matrix.py runs/<matrix_dir>
python3 scripts/summarize_llamacpp_matrix.py runs/<graph_compare_dir> --graph-compare
```

`llama-bench` does not expose per-token decode-first or P95 samples. If those
are measured manually, add them with:

```bash
python3 scripts/summarize_llamacpp_matrix.py runs/<matrix_dir> \
  --manual-metrics manual_metrics.csv
```

`manual_metrics.csv`:

```csv
preset,mode,decode_first_tok_s,decode_steady_p95_tok_s
short,bf16,,
short,oscar-int4,,
short,int4,,
```

## Accuracy

Benchmarks: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME 25.
This llama.cpp branch compares BF16, OSCAR INT4, and plain INT4 on Granite 4.0
1B through `llama-server`. This mirrors the SGLang branch accuracy structure,
with INT4 as the current llama.cpp delivery target; INT2 remains available only
for research comparisons.

Accuracy outputs are generated locally under `runs/<accuracy_dir>/`:

| Benchmark | Metric | Runner |
|---|---|---|
| **GPQA** | Score | `llama-eval.py` |
| **GSM8K** | Accuracy | `llama-eval.py` |
| **MATH-500** | Score | `llama-eval.py` |
| **HumanEval** | Pass@1/2/5 | `llama-eval.py` with `human-eval` grader |
| **LCB V6** | Pass@1 | official LiveCodeBench runner through `llama-server` |
| **AIME25** | Score | `llama-eval.py` (`opencompass/AIME2025`) |

AIME25 is loaded from `opencompass/AIME2025`, configs `AIME2025-I` and
`AIME2025-II`, split `test`, for 30 total examples. Re-run the same
`OUT_DIR=...` command to resume; completed JSON files and LCB variant markers
are skipped by default.

For setup and rerun commands, see [How to rerun accuracy](#how-to-rerun-accuracy).

### How to rerun accuracy

Install optional eval dependencies:

```bash
python3 -m pip install datasets pandas requests tqdm
python3 -m pip install git+https://github.com/openai/human-eval.git
```

For LiveCodeBench v6, use a separate environment if needed:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git third_party/LiveCodeBench
python3 -m pip install -e third_party/LiveCodeBench
```

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

Run a small smoke before the full suite:

```bash
OUT_DIR=runs/granite_accuracy_smoke_$(date +%Y%m%d_%H%M%S) \
GPQA_N_CASES=10 GSM8K_N_CASES=10 MATH500_N_CASES=10 AIME25_N_CASES=10 \
HUMANEVAL_N_CASES=20 HUMANEVAL_SAMPLES=1 RUN_LCB=0 \
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

Compare or inspect outputs:

```bash
cat runs/<granite_accuracy_dir>/accuracy_comparison.md
cat runs/<granite_accuracy_dir>/non_lcb/summary.csv
ls runs/<granite_accuracy_dir>/lcb_v6/raw/<variant>/lcb_output/
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

### Accuracy troubleshooting

- Overseas or unrestricted networks should use the official Hugging Face Hub
  directly. Do not set `HF_ENDPOINT` unless you intentionally want a mirror.
- In China or restricted networks, unset mirror endpoints that cause stale data,
  then set a local proxy and persistent cache before loading datasets:

```bash
unset HF_ENDPOINT
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
export ALL_PROXY=http://127.0.0.1:10808
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export HF_HOME=/dfs/data/tmp/hf
export HF_DATASETS_CACHE=/dfs/data/tmp/hf/datasets
```

- `Idavidrein/gpqa` is gated. Use an authenticated Hugging Face cache/token, or
  the CSV fallback in the vendored `llama-eval.py` path.
- If `lcb_runner` imports fail with `ModuleNotFoundError: anthropic`, install the
  missing optional dependency in the LiveCodeBench Python environment.
- If LiveCodeBench reports `KeyError: 'granite-4.0-1b-base'`, keep
  `LCB_MODEL_NAME=granite-4.0-1b-base`; the wrapper resolves it to an available
  OpenAI-compatible LiveCodeBench adapter such as `gpt-3.5-turbo-0125`, while
  requests still go to local `llama-server`.
- If Hugging Face `datasets` raises `Dataset scripts are no longer supported` for
  `livecodebench/code_generation_lite`, use `datasets<4` for the LiveCodeBench
  environment.
- The wrapper defaults to `baseline_bf16,oscar_int4,plain_int4`. Older local runs
  may have used INT2 defaults; start a fresh `OUT_DIR` after pulling this branch.

## Verification

Run the no-GPU consistency checks:

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

This checks archived summaries, static q2 CUDA boundaries, command generation,
and benchmark harness safety defaults. It does not run GPU benchmarks by
default.

Useful lightweight checks:

```bash
python3 scripts/check_execution_entrypoints.py
bash -n scripts/bench_32k_llamacpp_kv.sh scripts/run_llamacpp_accuracy_suite.sh scripts/run_granite_accuracy_full.sh
python3 -m py_compile scripts/summarize_llamacpp_matrix.py scripts/summarize_llamacpp_accuracy_suite.py scripts/summarize_granite_accuracy_full.py
```

## Limitations

- The current successful llama.cpp long-context delivery result is INT4, not
  INT2.
- The q2/INT2 32K path is intentionally guarded because the archived run timed
  out.
- Full accuracy runs can be slow on an 8GB laptop GPU.
- HumanEval and LiveCodeBench execute generated code during grading; the scripts
  require explicit acknowledgements before real runs.
- Rotation files under `rotation/**/rotations/` are treated as local artifacts
  and are ignored by the parent repo.
