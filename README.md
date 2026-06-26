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

Archived 32K llama.cpp matrix:

```bash
runs/llamacpp_32k_kv_matrix_current/combined.md
runs/llamacpp_32k_kv_matrix_current/combined.csv
```

Summary from the archived run:

| variant | prompt | status | KV | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---|---|---:|---:|---:|---:|
| `baseline_bf16` | 32768 | ok | `bf16/bf16` | 2560.0 | 6160 | 2486.4 | 41.6 |
| `oscar_int4` | 32768 | ok | `q4_0/q4_0` | 720.0 | 4324 | 2533.8 | 39.2 |
| `plain_int4` | 32768 | ok | `q4_0/q4_0` | 720.0 | 4324 | 2265.0 | 41.0 |
| `plain_int2` | 16384 | ok | `q2_0/q2_0` | 240.0 | 3792 | 180.0 | 44.1 |
| `oscar_int2` | 16384 | ok | `q2_0/q2_0` | 240.0 | 3796 | 183.7 | 28.0 |
| `oscar_int2` | 32768 | failed | `q2_0/q2_0` | 480.0 | 4036 | | |

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

The llama.cpp accuracy wrappers use `llama-server` and the same delivery variant
set by default: BF16, OSCAR INT4, and plain INT4. INT2 variants remain available
for research comparisons, but they are not the current delivery target.

Supported tasks:

| Benchmark | Metric | Runner |
|---|---|---|
| GPQA | Score | `llama-eval.py` |
| GSM8K | Accuracy | `llama-eval.py` |
| MATH500 | Score | `llama-eval.py` |
| HumanEval | Pass@1 | `llama-eval.py` with `human-eval` grader |
| AIME25 | Score | `llama-eval.py` (`aime2025`) |
| LCB V6 | Pass@1 | official LiveCodeBench runner through llama-server |

Smoke run:

```bash
OUT_DIR=runs/llamacpp_accuracy_smoke_$(date +%Y%m%d_%H%M%S) \
VARIANTS=baseline_bf16,oscar_int4,plain_int4 \
DATASETS=gpqa,gsm8k,math500,aime2025 \
GPQA_N_CASES=10 \
GSM8K_N_CASES=10 \
MATH500_N_CASES=10 \
AIME25_N_CASES=10 \
DRY_RUN=0 \
ACK_EVAL=1 \
  scripts/run_llamacpp_accuracy_suite.sh
```

Full non-LCB suite:

```bash
OUT_DIR=runs/llamacpp_accuracy_full_$(date +%Y%m%d_%H%M%S) \
VARIANTS=baseline_bf16,oscar_int4,plain_int4 \
DATASETS=gpqa,gsm8k,math500,humaneval,aime2025 \
GPQA_N_CASES=198 \
GSM8K_N_CASES=200 \
MATH500_N_CASES=500 \
HUMANEVAL_N_CASES=164 \
AIME25_N_CASES=60 \
DRY_RUN=0 \
ACK_EVAL=1 \
ALLOW_HUMANEVAL_EXEC=1 \
EVAL_TIMEOUT_SEC=0 \
  scripts/run_llamacpp_accuracy_suite.sh
```

Outputs:

```bash
runs/<accuracy_dir>/accuracy_comparison.md
runs/<accuracy_dir>/summary.csv
runs/<accuracy_dir>/raw/
runs/<accuracy_dir>/logs/
```

LiveCodeBench v6:

```bash
OUT_DIR=runs/llamacpp_lcb_v6_$(date +%Y%m%d_%H%M%S) \
VARIANTS=baseline_bf16,oscar_int2,plain_int2 \
LIVE_CODE_BENCH_ROOT=third_party/LiveCodeBench \
LCB_RELEASE=release_v6 \
LCB_N=1 \
DRY_RUN=0 \
ACK_EVAL=1 \
ALLOW_CODE_EXEC=1 \
  scripts/run_llamacpp_lcb_v6.sh
```

LCB outputs are copied under:

```bash
runs/<lcb_dir>/logs/
runs/<lcb_dir>/raw/<variant>/lcb_output/
```

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
bash -n scripts/bench_32k_llamacpp_kv.sh scripts/run_llamacpp_accuracy_suite.sh
python3 -m py_compile scripts/summarize_llamacpp_matrix.py scripts/summarize_llamacpp_accuracy_suite.py
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
