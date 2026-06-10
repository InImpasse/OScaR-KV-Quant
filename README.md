# OSCAR-KV-Quant

This repository is a local validation harness for [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR) (paper [arXiv:2605.17757](https://arxiv.org/abs/2605.17757)), with the main path validated on **Granite 4.0 1B** under WSL2 + RTX 5050 8GB.

Here, OSCAR means **Offline Spectral Covariance-Aware Rotation**. It is **not** the unrelated [ZunhaiSu/OScaR-KV-Quant](https://github.com/ZunhaiSu/OScaR-KV-Quant), which uses Omni-Scaled Canalized Rotation in a separate codebase.

## Current status

Validated path: **Granite 4.0 1B** on WSL2 + RTX 5050 8GB.

- **Speed/memory**: BF16, OSCAR INT2, and plain INT2 at prefill lengths 512–32K — see [Speed and memory](#speed-and-memory).
- **Accuracy**: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME 25 — see [Accuracy](#accuracy).
- **CUDA graph**: enabled for `oscar-int2`; `oscar-int8` / `oscar-int4` remain experimental — see [KV modes](#kv-modes) and [Limitations](#limitations).
- **Gemma 4 E2B**: early local tests were less promising on this 8GB setup; this repo keeps Granite as the main validated path and will revisit Gemma separately.

## Key takeaways

- CUDA graph is the primary OSCAR INT2 runtime path here; it improves steady decode throughput versus graph-off runs, with only modest whole-GPU peak-memory changes in the measured matrix.
- OSCAR INT2 substantially reduces the measured KV pool at long context. At 32K, the CUDA-graph-on run uses 512 MiB K+V pool versus 2990 MiB for BF16.
- Under the separate decode-through capacity criterion (`max_new_tokens=64`, HTTP <= 300s), OSCAR INT2 reaches 69632 tokens versus BF16's 38272-token allocatable KV limit, about **+82%** more context capacity. Plain INT2 reaches 70656 tokens (**+85%**), but with the accuracy collapse noted below.
- Whole-GPU peak memory falls much less than the KV pool because the peak also includes BF16 model weights, CUDA context, allocator reserves, workspaces, graph-capture buffers, and runtime overhead. To push total VRAM lower, a follow-up direction is the [`llamacpp-kv-harness`](https://github.com/InImpasse/OSCAR-KV-Quant/tree/llamacpp-kv-harness) branch, which tests llama.cpp + GGUF KV-cache formats.
- OSCAR INT2 preserves far more accuracy than plain INT2, which has the smallest KV pool but collapses on the accuracy suite. HumanEval Pass@1/Pass@2 remain the main OSCAR INT2 regressions.
- Results are local to Granite 4.0 1B on RTX 5050 8GB. They are not official OSCAR paper numbers and should not be compared directly to H100 / larger-model results.

## Requirements

- Python 3.12
- CUDA **12.8+** with `nvcc` (`scripts/check_env.sh` accepts any **nvcc ≥ 12.8**). Upstream docs still center on **12.8–12.9** for the vendored **cu129** stack; if FlashInfer/Triton JIT miscompiles, set `CUDA_HOME` to a toolkit that matches your PyTorch CUDA build.
- NVIDIA driver visible from WSL (`nvidia-smi`)
- Git submodules initialized
- `HF_TOKEN` for gated model downloads, if needed

The setup script creates `.venv-oscar-kv` and installs the vendored SGLang stack from `third_party/OSCAR`. The core project dependencies are pinned to the same stack: `torch==2.9.1`, `torchaudio==2.9.1`, and `transformers==5.3.0`. The broader SGLang install also expects `flashinfer_python==0.6.7.post3`, `sglang-kernel==0.4.1`, and `kernels==0.13.0`.

The upstream OSCAR project is [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR). This repository points `third_party/OSCAR` at the [InImpasse/OSCAR](https://github.com/InImpasse/OSCAR) fork through `.gitmodules`. The parent repo pins an exact submodule commit, so `git submodule update --init --recursive` checks out that commit instead of following a branch.

## Quick start

From a fresh clone, use the [`sglang-kv-harness`](https://github.com/InImpasse/OSCAR-KV-Quant/tree/sglang-kv-harness) branch if it is not already checked out:

```bash
git clone https://github.com/InImpasse/OSCAR-KV-Quant.git
cd OSCAR-KV-Quant
git fetch origin
git switch sglang-kv-harness
```

```bash
git submodule update --init --recursive
./scripts/check_env.sh
./scripts/setup_env_uv.sh
./scripts/download_models.sh granite

./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --kv-cache-dtype bf16

./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes bf16,int2 \
  --results-dir results/smoke
```

The quick-start path validates Granite and only downloads `checkpoints/granite-4.0-1b-base`. Run `./scripts/download_models.sh` with no arguments to download all configured models, or `./scripts/download_models.sh gemma4` to fetch only the Gemma 4 E2B checkpoint.

## Environment setup details

Use the uv-based setup script for a reproducible environment. It installs Python 3.12 when needed, creates `.venv-oscar-kv`, installs this package, installs the vendored OSCAR/SGLang stack, and finishes with a basic import probe.

```bash
# Do not run this script with sudo.
./scripts/setup_env_uv.sh
```

Wrapper scripts under `scripts/` automatically use `.venv-oscar-kv`, so manual activation is optional. For interactive debugging, activate it directly:

```bash
source .venv-oscar-kv/bin/activate
hash -r
```

If commands are missing or the environment was created before dependency pins changed, rerun setup:

```bash
./scripts/setup_env_uv.sh
```

If the virtualenv already exists and only the vendored OSCAR/SGLang install needs repair, use the lighter fallback:

```bash
./scripts/install_sglang_os.sh
```

If CUDA is not discoverable from the default install path, pass the matching toolkit explicitly:

```bash
./scripts/setup_env_uv.sh --cuda-home /usr/local/cuda-12.9
./scripts/probe.sh --cuda-home /usr/local/cuda-12.9 --try-dummy-server
```

Run unit tests before long GPU jobs:

```bash
./scripts/test.sh
./scripts/test.sh --run-server-tests  # optional local SGLang server test
```

## Reproducing the baseline

Start by verifying the host and the Granite BF16 path:

```bash
./scripts/check_env.sh

./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --kv-cache-dtype bf16 \
  --timeout-s 180
```

In a healthy probe, CUDA is available, the RTX 5050 is detected, and the output includes `sglang_import_ok: true` and `flashinfer_import_ok: true`.

If `--try-model-server` fails with `RuntimeError: Cannot find any model weights` even though `model.safetensors` exists, the directory often contains a **shard-style** `model.safetensors.index.json` (referencing `model-00001-of-00002.safetensors`, …) while only a **single** consolidated `model.safetensors` is on disk. Re-run `./scripts/download_models.sh granite` so the script can move the stale index aside, or manually rename/remove `model.safetensors.index.json` once you confirm the consolidated file is complete.

Then probe INT2:

```bash
./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --try-int2 \
  --timeout-s 180
```

If INT2 hits FA3 issues on `sm_120`, use Triton attention:

```bash
./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --try-int2 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --timeout-s 180
```

Run short and medium Granite baselines before attempting 32K:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes bf16,int2 \
  --request-api completions \
  --results-dir results/granite_short

./scripts/bench.sh \
  --profile granite \
  --preset medium \
  --modes bf16,int2 \
  --request-api completions \
  --results-dir results/granite_medium
```

After rotation files are available, run the OSCAR INT2 path:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes oscar-int2 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --request-api completions \
  --results-dir results/granite_oscar_short
```

Benchmarks write CSV and Markdown reports under `results/`. The fields to check first are `server_ok`, `error`, `decode_toks_per_sec`, `decode_steady_median_tok_s`, `peak_mib_delta`, `kv_theory_bf16_gib`, `kv_theory_selected_gib`, `kv_k_size_gb`, and `kv_v_size_gb`.

For the full speed/memory matrix across 512, 2K, 8K, 16K, and 32K, use `bench_matrix.sh` or `cuda_graph_compare_matrix.sh` in [Speed and memory](#speed-and-memory).

## Official OSCAR vs this repo

| | Official OSCAR | This repo |
|---|---|---|
| **Hardware** | H100 80GB | RTX 5050 8GB (`sm_120`) |
| **Models** | Qwen3, Gemma 4 12B, GLM, MiniMax, ... | Granite 4.0 1B baseline |
| **Rotations** | [OSCAR-RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo) | Local Granite calibration under `rotation/granite-4.0-1b/` |
| **Accuracy protocol** | GPQA, HumanEval, LCB v6, AIME 25, MATH-500; five repeats | Same wrapper shape, smaller local subsets as GPU time allows |
| **KV compression** | Paper reports about 2.28 BPE vs BF16 16.00 at long context | Local 32K OSCAR INT2 theoretical KV is about 15% of BF16 |

This repository does **not** try to reproduce the official paper numbers on Granite. It keeps the comparison structure the same: BF16 baseline, OSCAR INT2, plain INT2, and the same metric family, but with a different model and different hardware.

### Local SGLang/OSCAR fork notes

The vendored OSCAR/SGLang tree is not a vanilla upstream checkout. For the local RTX 5050 / `sm_120` path, this repo uses the pinned `InImpasse/OSCAR` fork and wrapper defaults that keep INT2 testing usable on consumer Blackwell hardware:

- `int2` and `oscar-int2` default to Triton prefill/decode backends on `sm_120`, because the current FA3/FA4 dense INT2 prefill path is not usable there.
- The fork contains an `SM120` fallback in the Triton attention path: it uses SDPA over already-dequantized per-request K/V for the incompatible dense prefill step, while keeping the quantized KV cache path testable.
- Long-context wrapper runs cap `--max-total-tokens` for fair BF16/INT2 KV-pool comparisons on an 8GB GPU.

## KV modes

| Mode | SGLang `--kv-cache-dtype` | Notes |
|---|---|---|
| **`bf16`** | `bf16` | Full-precision KV baseline |
| **`oscar-int2`** | `int2` | Mixed-precision windows plus rotation `.pt` files |
| **`int2`** | `int2` | Plain INT2, no rotation |
| **`oscar-int8` / `oscar-int4`** | `int8` / `int4` | Experimental rotated integer KV modes; run with `--disable-oscar-cuda-graph` in the current runtime |
| **`fp8` / `fp4`** | `fp8_e4m3` / `fp4_e2m1` | Floating low-bit KV comparison modes |
| **`int8` / `int4`** | `int8` / `int4` | Integer KV modes; check runtime pool logs before interpreting memory |

Granite is BF16-trained. Forced FP16 compute collapses outputs, so use `bf16` or `auto` as the full-precision baseline.

## Speed and memory

Bench runs target a **prefill** length in **input tokens**. The prompt is synthetic text built from the model tokenizer; see `_build_prefill_text` in `src/oscar_kv_quant/bench.py`. Decode length is configured separately with `--max-new-tokens` and defaults to 64 in the baseline tables.

The CUDA graph comparison below was captured on RTX 5050 with `max_new_tokens=64` and the completions API. It reports BF16, OSCAR INT2, and plain INT2 with CUDA graph enabled and disabled. Graph-on is the primary OSCAR INT2 path; graph-off is kept for isolation and debugging.

For rerun commands, preset definitions, and metric details, see [How to rerun and read this matrix](#how-to-rerun-and-read-this-matrix) after the tables.

### CUDA graph on

Preset mapping: `512` = `short`, `2K` = `medium`, `8K` = `long`, `16K` = `16k`, `32K` = `32k`.

#### Decode first (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 4.98 | 5.81 | +17% | 4.72 | -5% |
| **2K** | 4.65 | 4.45 | -4% | 4.51 | -3% |
| **8K** | 3.88 | 4.18 | +8% | 3.51 | -10% |
| **16K** | 2.94 | 2.59 | -12% | 3.00 | +2% |
| **32K** | 1.58 | 1.78 | +13% | 1.71 | +8% |

#### Steady (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 76.98 | 53.53 | -30% | 56.55 | -27% |
| **2K** | 71.69 | 52.03 | -27% | 67.43 | -6% |
| **8K** | 64.64 | 46.76 | -28% | 52.31 | -19% |
| **16K** | 52.84 | 40.71 | -23% | 53.57 | +1% |
| **32K** | 34.56 | 39.27 | +14% | 42.27 | +22% |

#### Peak (MiB, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 3802 | 3839 | +1% (+37 MiB) | 3799 | 0% (-3 MiB) |
| **2K** | 3875 | 4000 | +3% (+125 MiB) | 3836 | -1% (-39 MiB) |
| **8K** | 4392 | 4009 | -9% (-383 MiB) | 3829 | -13% (-563 MiB) |
| **16K** | 5007 | 4419 | -12% (-588 MiB) | 4323 | -14% (-684 MiB) |
| **32K** | 6607 | 5937 | -10% (-670 MiB) | 5921 | -10% (-686 MiB) |

#### KV pool K+V (MiB, measured, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 123 | 102 | -17% (-21 MiB) | 20 | -84% (-103 MiB) |
| **2K** | 246 | 123 | -50% (-123 MiB) | 41 | -83% (-205 MiB) |
| **8K** | 737 | 184 | -75% (-553 MiB) | 82 | -89% (-655 MiB) |
| **16K** | 1372 | 287 | -79% (-1085 MiB) | 164 | -88% (-1208 MiB) |
| **32K** | 2990 | 512 | -83% (-2478 MiB) | 369 | -88% (-2621 MiB) |

#### Prefill (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 382 | 237 | -38% | 293 | -23% |
| **2K** | 988 | 813 | -18% | 1586 | +61% |
| **8K** | 3121 | 2579 | -17% | 2409 | -23% |
| **16K** | 3039 | 2037 | -33% | 2795 | -8% |
| **32K** | 1676 | 2066 | +23% | 1815 | +8% |

#### P95 (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 82.38 | 60.72 | -26% | 64.28 | -22% |
| **2K** | 77.94 | 59.29 | -24% | 73.92 | -5% |
| **8K** | 67.19 | 56.30 | -16% | 59.52 | -11% |
| **16K** | 57.68 | 52.55 | -9% | 64.50 | +12% |
| **32K** | 42.69 | 51.09 | +20% | 58.85 | +38% |

### CUDA graph off

#### Decode first (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 5.52 | 5.06 | -8% | 4.85 | -12% |
| **2K** | 4.93 | 4.51 | -9% | 4.67 | -5% |
| **8K** | 4.05 | 3.55 | -12% | 3.65 | -10% |
| **16K** | 3.01 | 3.03 | +1% | 2.64 | -12% |
| **32K** | 1.62 | 1.70 | +5% | 1.96 | +21% |

#### Steady (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 46.02 | 27.07 | -41% | 34.61 | -25% |
| **2K** | 44.72 | 27.16 | -39% | 33.76 | -25% |
| **8K** | 45.15 | 26.62 | -41% | 32.63 | -28% |
| **16K** | 40.12 | 24.13 | -40% | 32.60 | -19% |
| **32K** | 32.90 | 21.77 | -34% | 26.34 | -20% |

#### Peak (MiB, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 3806 | 3818 | 0% (+12 MiB) | 3806 | 0% (0 MiB) |
| **2K** | 3824 | 3944 | +3% (+120 MiB) | 3806 | 0% (-18 MiB) |
| **8K** | 4344 | 3978 | -8% (-366 MiB) | 3806 | -12% (-538 MiB) |
| **16K** | 4986 | 4398 | -12% (-588 MiB) | 4302 | -14% (-684 MiB) |
| **32K** | 6586 | 5916 | -10% (-670 MiB) | 5900 | -10% (-686 MiB) |

#### KV pool K+V (MiB, measured, lower better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 123 | 102 | -17% (-21 MiB) | 20 | -84% (-103 MiB) |
| **2K** | 246 | 123 | -50% (-123 MiB) | 41 | -83% (-205 MiB) |
| **8K** | 737 | 184 | -75% (-553 MiB) | 82 | -89% (-655 MiB) |
| **16K** | 1372 | 287 | -79% (-1085 MiB) | 164 | -88% (-1208 MiB) |
| **32K** | 2990 | 512 | -83% (-2478 MiB) | 369 | -88% (-2621 MiB) |

#### Prefill (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 230 | 119 | -48% | 175 | -24% |
| **2K** | 704 | 381 | -46% | 756 | +7% |
| **8K** | 3053 | 2870 | -6% | 2466 | -19% |
| **16K** | 2874 | 2154 | -25% | 1985 | -31% |
| **32K** | 1657 | 1883 | +14% | 2154 | +30% |

#### P95 (tok/s, higher better)

| Length (tokens) | BF16 | OSCAR INT2 | Δ vs BF16 | plain INT2 | Δ vs BF16 |
|---:|---:|---:|---:|---:|---:|
| **512** | 49.11 | 28.18 | -43% | 37.60 | -23% |
| **2K** | 48.40 | 28.24 | -42% | 37.49 | -23% |
| **8K** | 48.64 | 27.82 | -43% | 32.72 | -33% |
| **16K** | 43.41 | 26.54 | -39% | 38.50 | -11% |
| **32K** | 39.80 | 27.71 | -30% | 36.14 | -9% |

- **Note (8K OSCAR, graph off):** the request layer aborts with `Connection reset by peer`; steady/P95 cells for `8K` are **last-window partial metrics** before the crash.

### Derived: toggling CUDA graph (on vs off)

Steady decode **speedup** is graph-on steady ÷ graph-off steady. **Peak Δ** is graph-on peak MiB − graph-off peak MiB from whole-GPU `nvidia-smi` sampling.

| Prefill | BF16 steady × | OSCAR steady × | plain INT2 steady × | BF16 peak Δ | OSCAR peak Δ | plain INT2 peak Δ |
|---:|---:|---:|---:|---:|---:|---:|
| **512** | 1.67× | 1.98× | 1.63× | -4 MiB | +21 MiB | -7 MiB |
| **2K** | 1.60× | 1.92× | 2.00× | +51 MiB | +56 MiB | +30 MiB |
| **8K** | 1.43× | 1.76× | 1.60× | +48 MiB | +31 MiB | +23 MiB |
| **16K** | 1.32× | 1.69× | 1.64× | +21 MiB | +21 MiB | +21 MiB |
| **32K** | 1.05× | 1.80× | 1.60× | +21 MiB | +21 MiB | +21 MiB |

CUDA graph improves steady decode when the graph-off run completes. OSCAR **8K graph off** still hits the mixed-KV idle/crash path, so that row is a partial timing snapshot. Whole-GPU peak changes only modestly when graph is enabled (~+21 MiB at 16K/32K in this run).

### How to rerun and read this matrix

Raw outputs live under `results/cuda_graph_compare_matrix/<TAG>/`. Throughput and memory numbers come from `oscar-kv-bench` CSV fields / SGLang server logs (see `src/oscar_kv_quant/log_metrics.py`). Rows are grouped by **prefill length** (input tokens). **Δ vs BF16** is `(mode - BF16) / BF16`; Peak and KV rows also show absolute MiB delta.

**Named presets** — these are the valid `--preset` / `--presets` labels. The CLI rejects any other label:

| Preset | Prefill tokens |
|--------|---------------:|
| **`short`** | 512 |
| **`medium`** | 2048 |
| **`long`** | 8192 |
| **`16k`** | 16384 |
| **`32k`** | 32768 |

- **Single bench**: `./scripts/bench.sh` / `oscar-kv-bench` uses `--preset <name>` and defaults to `short`.
- **Preset matrix**: `./scripts/bench_matrix.sh` uses `--presets short,medium,long,16k,32k` as a comma-separated list. Any subset and order are valid.
- **Custom length**: `--prefill-tokens N` overrides the preset for that run. This is useful for smoke tests or lengths not listed above. Named presets remain defined in `PRESET_TOKENS` in `src/oscar_kv_quant/bench.py`.

For `16k` and `32k`, the matrix wrapper caps `--max-total-tokens` at `17408` and `38272` respectively, so BF16 and INT2 allocate comparable KV pools. If running `oscar-kv-bench` directly at 32K, pass `--max-total-tokens 38272`; uncapped INT2 KV pools can auto-size much larger and are not directly comparable.

| Metric | Meaning | Better |
|---|---|---|
| **Decode first** (tok/s) | Throughput of the **first** logged decode step (`decode_first_tok_s`). Often reflects cold-start / graph-capture effects. | Higher |
| **Steady** (tok/s) | Median decode throughput after dropping the first step and low flush/scheduler outliers (`decode_steady_median_tok_s`). Main sustained decode speed. | Higher |
| **P95** (tok/s) | 95th percentile of those steady decode samples (`decode_steady_p95_tok_s`). Tail decode speed; usually ≥ Steady. | Higher |
| **Peak** (MiB) | Whole-GPU peak memory from `nvidia-smi` (`peak_mib_total`), including weights, CUDA context, workspaces, graph capture, and KV. | Lower |
| **KV pool K+V** (MiB, measured) | K+V allocator size parsed from the SGLang `KV Cache is allocated...` log line (`kv_k_size_gb` + `kv_v_size_gb`), not a theoretical estimate. | Lower |
| **Prefill** (tok/s) | Median prompt-ingest throughput while filling the input into KV cache (`prefill_median_tok_s`; first two prefill samples skipped when enough data). | Higher |

To rerun and summarize this matrix:

```bash
./scripts/cuda_graph_compare_matrix.sh \
  --tag "$(date +%Y%m%d)" \
  --presets short,medium,long,16k,32k \
  --modes bf16,oscar-int2,int2

python scripts/summarize_cuda_graph_matrix.py \
  results/cuda_graph_compare_matrix/<TAG>
```

To run the preset matrix without the graph-on/off split:

```bash
./scripts/bench_matrix.sh \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --tag "$(date +%Y%m%d)" \
  --presets short,medium,long,16k,32k \
  --modes bf16,oscar-int2,int2
```

### RTX 5050 long-prefill limits

The local RTX 5050 8GB long-prefill limit is defined as a **complete** single-request run with `max_new_tokens=1`, `mem_fraction_static=0.88`, Triton prefill/decode backends for INT2 modes, and an empty GPU before launch. Timeout-only or hung runs are counted as failures. Raw outputs are under `results/limit_prefill_retest_5050/`.

| Mode | 6 min limit | Next tested | 10 min limit | Next tested |
|---|---:|---:|---:|---:|
| **BF16** | 41952 | 41968 timeout | 41952 | 41968 timeout |
| **OSCAR INT2** | 65536 | 69632 at 379.05s | 80896 | 81920 at 601.84s |
| **plain INT2** | 65536 | 69632 at 384.74s | 80896 | 81920 at 606.25s |

For decode-through capacity, use a separate **HTTP <= 300s** single-request criterion with `max_new_tokens=64`, `mem_fraction_static=0.88`, Triton prefill/decode backends for INT2 modes, and an empty GPU before launch. BF16 is capped by actual SGLang KV allocation: requesting 38400 tokens still profiles only a 38272-token pool. Raw outputs are under `results/kv_capacity_*/`.

| Mode | Max successful KV capacity | Next tested failure | HTTP elapsed at max success | Notes |
|---|---:|---:|---:|---|
| **BF16** | 38272 | 38400 requested -> 38272 actual | <=300s at 38272 | Actual allocatable KV pool limit is 38272. |
| **OSCAR INT2** | 69632 | 70656 timeout | 274.54s | More headroom at 69632, but 70656/71680/73728 all timed out. |
| **plain INT2** | 70656 | 71680 timeout | 298.94s | Highest tested success; very close to the 300s boundary. |

Gate a 32K CSV against BF16 and plain INT2:

```bash
./scripts/regression_gate.sh --scenario balanced results/granite_bench_matrix/*/32k/bench_*.csv
```

Scenarios: `memory`, `balanced` (default), and `speed`.

To search for the largest stable prefill length on the local GPU:

```bash
python scripts/probe_max_prefill.py \
  --modes bf16,oscar-int2,int2 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --cap 120000 \
  --triton-for-int2-modes
```

For OSCAR INT2 runtime configuration sweeps:

```bash
./scripts/config_sweep.sh \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --prefill-tokens 32768 \
  --results-dir results/config_sweep
```

## Accuracy

Benchmarks: GPQA, GSM8K, MATH-500, HumanEval, LiveCodeBench v6, and AIME 25. Each compares BF16, OSCAR INT2, and plain INT2 on Granite 4.0 1B.

Accuracy comparison:

| Benchmark | Metric | BF16 | OSCAR INT2 | Δ vs BF16 | Plain INT2 | Δ vs BF16 |
|---|---|---:|---:|---:|---:|---:|
| **GPQA** | Score | 23.74 | 24.24 | **+0.50 pt** | 15.66 | **-8.08 pt** |
| **GSM8K** | Accuracy | 56.0 | 54.5 | **-1.5 pt** | 3.0 | **-53.0 pt** |
| **MATH500** | Score | 7.40 | 7.20 | **-0.20 pt** | 0.20 | **-7.20 pt** |
| **LCB V6** | Pass@1 | 7.87 | 6.92 | **-0.95 pt** | 0.00 | **-7.87 pt** |
| **HumanEval** | Pass@1 | 32.93 | 12.68 | **-20.25 pt** | 0.00 | **-32.93 pt** |
| **HumanEval** | Pass@2 | 33.66 | 19.88 | **-13.78 pt** | 0.00 | **-33.66 pt** |
| **HumanEval** | Pass@5 | 34.76 | 32.93 | **-1.83 pt** | 0.00 | **-34.76 pt** |
| **AIME25** | Score | 0.00 | 0.00 | — | 0.00 | — |

OSCAR INT2 stays close to BF16 on GPQA, GSM8K, MATH500, HumanEval Pass@5, and LCB v6, while plain INT2 collapses across the accuracy suite. HumanEval Pass@1 and Pass@2 remain the main OSCAR INT2 regressions.

For setup and rerun commands, see [How to rerun accuracy](#how-to-rerun-accuracy).

### How to rerun accuracy

Install optional eval dependencies:

```bash
bash scripts/setup_eval_suite.sh
```

Run the full Granite suite across BF16, OSCAR INT2, and plain INT2:

```bash
bash rotation/granite-4.0-1b/eval_accuracy_suite.sh \
  --gpqa-num-examples 198 \
  --gsm8k-num-questions 200 \
  --humaneval-num-examples 164 \
  --math-num-examples 500 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations
```

Pass `0` to skip an expensive benchmark, for example `--math-num-examples 0`. Each eval script also has `--help`.

To rerun only missing benchmarks in an existing results tree:

```bash
./scripts/run_granite_accuracy_fill.sh --only all
```

Compare existing run directories:

```bash
oscar-kv-accuracy-compare \
  --bf16 path/to/bf16_run \
  --int2 path/to/int2_run \
  --oscar-int2 path/to/oscar_run
```

## Rotation workflow

Official OSCAR rotations are model-specific and cannot be reused for Granite. The local Granite calibration path uses GPQA, 30K calibration tokens, and group size 128 by default.

```bash
bash rotation/granite-4.0-1b/save_qkv_granite.sh
bash rotation/granite-4.0-1b/compute_rotation.sh

oscar-kv-rotation-gate \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations
```

Serving expects exactly these files in `ROT_DIR`:

- `k_rotation_qqt_r_h_pbr.pt`
- `v_rotation_sst_r_h_pbr.pt`

For a smoke calibration on an 8GB GPU:

```bash
bash rotation/granite-4.0-1b/save_qkv_granite.sh --calib-profile smoke
bash rotation/granite-4.0-1b/compute_rotation.sh --calib-profile smoke --allow-weak-calibration
```

## Testing and troubleshooting

`./scripts/test.sh` is primarily a unit-test runner, not a full GPU benchmark suite. By default, it runs `python -m unittest discover -s tests` inside `.venv-oscar-kv`, then performs a lightweight CLI smoke pass over `probe.sh --help`, `bench.sh --help`, `probe.sh`, and a Granite BF16/INT2 `bench.sh --dry-run`. It covers this repository's root `tests/` directory and wrapper entry points; it does not run the large upstream test suites under `third_party/OSCAR/`, does not start a model server by default, and does not validate long-context throughput, VRAM peaks, or accuracy.

`scripts/test.sh` options:

| Command | What it runs | When to use |
|---|---|---|
| `./scripts/test.sh` | Root `tests/` unittest discovery plus wrapper/CLI smoke checks | Default preflight before code changes or local runs |
| `./scripts/test.sh --unit-only` | Root `tests/` unittest discovery only | Fastest check; no CLI smoke |
| `./scripts/test.sh --skip-cli-smoke` | Same unittest discovery, but skips wrapper/CLI smoke | Use when only Python unit coverage matters |
| `./scripts/test.sh tests/test_bench_helpers.py` | A single test file or unittest selector | Focused debugging |
| `./scripts/test.sh --run-integration` | Default tests plus optional SGLang dummy-server probe | Local integration check; may use GPU/ports |
| `./scripts/test.sh --run-server-tests --test-port 31992` | Same integration check on a chosen port | Avoid stale port conflicts |

Use the optional integration layer when you want a local SGLang dummy-server check. For long-context throughput, VRAM peaks, CUDA graph behavior, and accuracy, use the benchmark and eval commands as acceptance tests.

Acceptance-test examples: `probe.sh --try-int2` for a real SGLang server path, `bench.sh` / `bench_matrix.sh` for CUDA graph, decode speed, and VRAM peaks, and the Granite eval scripts for GPQA / HumanEval accuracy.

Test file index:

| Test file | Purpose |
|---|---|
| `tests/test_bench_helpers.py` | Unit tests for bench mode mapping, server command construction, OSCAR env vars, and KV estimate helpers. |
| `tests/test_cli_dry_run.py` | Dry-run coverage for `oscar_kv_quant.bench` report generation without starting SGLang. |
| `tests/test_gpu_cuda_optional.py` | Optional CUDA, BF16, SGLang, and FlashInfer environment checks. |
| `tests/test_kv_estimate.py` | Unit tests for BF16, OSCAR INT2, and plain INT2 KV byte estimates. |
| `tests/test_log_metrics.py` | Unit tests for parsing SGLang server logs into bench metrics. |
| `tests/test_longrun_gate.py` | Unit tests for long-run speed, memory, stability, and log-cleanliness gates. |
| `tests/test_probe_helpers.py` | Unit tests for probe log, health-check, status, and model-name helpers. |
| `tests/test_profiles.py` | Unit tests for model profile and KV geometry parsing. |
| `tests/test_regression_gate.py` | Unit tests for bench CSV regression gate scenarios. |
| `tests/test_rotation_gate.py` | Unit tests for rotation metadata and accuracy-drop gates. |
| `tests/test_sglang_dump_writer.py` | Optional test for the vendored SGLang async Q/K/V dump writer. |
| `tests/test_sglang_server_optional.py` | Optional dummy SGLang server probe test. |

Environment-dependent tests stay in the suite because they protect the local CUDA/SGLang validation path and skip cleanly when prerequisites are missing.

## Repository layout

```text
scripts/             setup, probe, bench, gates, sweeps
src/oscar_kv_quant/  Python CLIs and shared helpers
benchmarks/          granite_*_baseline.json + RUN_LOG.md
rotation/            per-model calibration and eval wrappers
third_party/OSCAR/   vendored OSCAR submodule
results/             raw bench/eval outputs, gitignored
runs/                local run artifacts, gitignored
```

## Script index

| Command or script | Purpose |
|---|---|
| `scripts/check_env.sh` | Validate local CUDA/Python basics |
| `scripts/setup_env_uv.sh` | Create `.venv-oscar-kv` and install dependencies |
| `scripts/install_sglang_os.sh` | Fallback editable install for the vendored OSCAR/SGLang tree |
| `scripts/download_models.sh` | Download all models or one selected model |
| `scripts/probe.sh` / `oscar-kv-probe` | Probe SGLang KV dtype support |
| `scripts/probe_max_prefill.py` | Binary-search the largest successful Granite prefill length per KV mode |
| `scripts/bench.sh` / `oscar-kv-bench` | Single preset/mode bench |
| `scripts/bench_matrix.sh` | Preset x mode bench matrix |
| `scripts/cuda_graph_compare_matrix.sh` | CUDA graph on/off matrix for BF16, OSCAR INT2, and plain INT2 |
| `scripts/summarize_cuda_graph_matrix.py` | Summarize CUDA graph matrix CSVs into README-style tables |
| `scripts/regression_gate.sh` / `oscar-kv-regression-gate` | CSV gate vs BF16/INT2 |
| `scripts/config_sweep.sh` / `oscar-kv-config-sweep` | Wrapper for configuration sweeps |
| `scripts/setup_eval_suite.sh` | Install optional accuracy-eval dependencies |
| `scripts/run_granite_accuracy_fill.sh` | Rerun only missing benchmarks in an existing accuracy results tree |
| `scripts/test.sh` | Run root unit tests and lightweight CLI smoke checks |
| `scripts/lib/*.sh` | Internal shared shell helpers for repo paths, runtime caches, and eval CLI parsing |
| `oscar-kv-rotation-gate` | Validate rotation files |
| `oscar-kv-calibration-metrics` | Inspect calibration/rotation artifacts |
| `oscar-kv-longrun` / `oscar-kv-longrun-gate` | Longer-run bench and gating helpers |
| `rotation/granite-4.0-1b/eval_accuracy_suite.sh` | Batch Granite accuracy suite |
| `rotation/granite-4.0-1b/eval_*_granite*.sh` | Individual Granite eval wrappers |

## Limitations

- RTX 5050 / `sm_120` support is best-effort; upstream targets H100-class hardware.
- 8GB VRAM runs should start with `--preset short` and scale gradually.
- `int8` and `int4` need runtime pool-log checks before making storage claims.
- CUDA graph support for `oscar-int8` and `oscar-int4` is not implemented yet. These modes require `--disable-oscar-cuda-graph` because dynamic scale calculation uses GPU-to-CPU sync (`.item()`) during KV writes, which breaks CUDA graph capture at SGLang startup. A proper fix likely needs a graph-safe quantized KV write path, such as Triton/CUDA kernels that compute/store scales on device or a separate precomputed-scale design; simply removing `.item()` would risk incorrect dequantization or stale per-layer scales.
- The official five-benchmark, five-repeat protocol is expensive on a laptop GPU; scale `eval_accuracy_suite.sh` incrementally.
