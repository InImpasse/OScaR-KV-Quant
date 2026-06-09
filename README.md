# OSCAR-KV-Quant

This repository is a local validation harness for [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR) (paper [arXiv:2605.17757](https://arxiv.org/abs/2605.17757)), with the main path validated on **Granite 4.0 1B** under WSL2 + RTX 5050 8GB.

Here, OSCAR means **Offline Spectral Covariance-Aware Rotation**. It is **not** the unrelated [ZunhaiSu/OScaR-KV-Quant](https://github.com/ZunhaiSu/OScaR-KV-Quant), which uses Omni-Scaled Canalized Rotation in a separate codebase.

## Current status

- Granite 4.0 1B is the main validated path.
- **Speed/memory**: complete for 512 / 2K / 8K / 16K / 32K prefill × BF16 / plain INT2 / OSCAR INT2.
- **Accuracy**: GPQA (198) is complete. HumanEval (164) is complete for BF16 + plain INT2; OSCAR INT2 stopped at 61/164 because of a mixed-KV idle leak. MATH-500 has only a smoke run so far (N=20).
- CUDA graph status: `oscar-int2` is the validated graph-enabled OSCAR path. `oscar-int8` and `oscar-int4` are still experimental and currently require CUDA graph disabled; their dynamic PyTorch scale calculation calls GPU-to-CPU sync (`.item()`) during KV writes, which breaks CUDA graph capture during SGLang startup.
- LiveCodeBench v6 and AIME 25 have harnesses, but are not part of the current baseline.

## Requirements

- Python 3.12
- CUDA 12.8 or 12.9 with `nvcc`
- NVIDIA driver visible from WSL (`nvidia-smi`)
- Git submodules initialized
- `HF_TOKEN` for gated model downloads, if needed

The setup script creates `.venv-oscar-kv` and installs the vendored SGLang stack from `third_party/OSCAR`. The core project dependencies are pinned to the same stack: `torch==2.9.1`, `torchaudio==2.9.1`, and `transformers==5.3.0`. The broader SGLang install also expects `flashinfer_python==0.6.7.post3`, `sglang-kernel==0.4.1`, and `kernels==0.13.0`.

The upstream OSCAR project is [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR). This repository points `third_party/OSCAR` at the [InImpasse/OSCAR](https://github.com/InImpasse/OSCAR) fork through `.gitmodules`. The parent repo pins an exact submodule commit, so `git submodule update --init --recursive` checks out that commit instead of following a branch.

## Quick start

If you are using the [`sglang-kv-harness`](https://github.com/InImpasse/OSCAR-KV-Quant/tree/sglang-kv-harness) branch, switch to it right after cloning:

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
  --modes bf16,fp8,fp4,int2 \
  --request-api completions \
  --results-dir results/granite_short

./scripts/bench.sh \
  --profile granite \
  --preset medium \
  --modes bf16,fp8,fp4,int2 \
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

## Official OSCAR vs this repo

| | Official OSCAR | This repo |
|---|---|---|
| Hardware | H100 80GB | RTX 5050 8GB (`sm_120`) |
| Models | Qwen3, Gemma 4 12B, GLM, MiniMax, ... | Granite 4.0 1B baseline |
| Rotations | [OSCAR-RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo) | Local Granite calibration under `rotation/granite-4.0-1b/` |
| Accuracy protocol | GPQA, HumanEval, LCB v6, AIME 25, MATH-500; five repeats | Same wrapper shape, smaller local subsets as GPU time allows |
| KV compression | Paper reports about 2.28 BPE vs BF16 16.00 at long context | Local 32K OSCAR INT2 theoretical KV is about 15% of BF16 |

This repository does **not** try to reproduce the official paper numbers on Granite. It keeps the comparison structure the same: BF16 baseline, plain INT2, OSCAR INT2, and the same metric family, but with a different model and different hardware.

## KV modes

| Mode | SGLang `--kv-cache-dtype` | Notes |
|---|---|---|
| `bf16` | `bf16` | Full-precision KV baseline |
| `int2` | `int2` | Plain INT2, no rotation |
| `oscar-int2` | `int2` | Mixed-precision windows plus rotation `.pt` files |
| `oscar-int8` / `oscar-int4` | `int8` / `int4` | Experimental rotated integer KV modes; run with `--disable-oscar-cuda-graph` in the current runtime |
| `fp8` / `fp4` | `fp8_e4m3` / `fp4_e2m1` | Floating low-bit KV comparison modes |
| `int8` / `int4` | `int8` / `int4` | Integer KV modes; check runtime pool logs before interpreting memory |

Granite is BF16-trained. Forced FP16 compute collapses outputs, so use `bf16` or `auto` as the full-precision baseline.

## Speed and memory

Bench runs target a **prefill** length in **input tokens**. The prompt is synthetic text built from the model tokenizer; see `_build_prefill_text` in `src/oscar_kv_quant/bench.py`. Decode length is configured separately with `--max-new-tokens` and defaults to 64 in the baseline tables.

**Named presets** — these are the valid `--preset` / `--presets` labels. The CLI rejects any other label:

| Preset | Prefill tokens |
|--------|---------------:|
| `short` | 512 |
| `medium` | 2048 |
| `long` | 8192 |
| `16k` | 16384 |
| `32k` | 32768 |

- **Single bench**: `./scripts/bench.sh` / `oscar-kv-bench` uses `--preset <name>` and defaults to `short`.
- **Preset matrix**: `./scripts/bench_matrix.sh` uses `--presets short,medium,long,16k,32k` as a comma-separated list. Any subset and order are valid.
- **Custom length**: `--prefill-tokens N` overrides the preset for that run. This is useful for smoke tests or lengths not listed above. Named presets remain defined in `PRESET_TOKENS` in `src/oscar_kv_quant/bench.py`.

```bash
./scripts/bench_matrix.sh \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --tag "$(date +%Y%m%d)" \
  --presets short,medium,long,16k,32k \
  --modes bf16,int2,oscar-int2
```

For `16k` and `32k`, the wrappers cap `--max-total-tokens` at `17408` and `38272` respectively, so BF16 and INT2 allocate comparable KV pools.

The CUDA graph comparison below was captured on RTX 5050 with `max_new_tokens=64` and the completions API. It reports BF16, plain INT2, and OSCAR INT2 with CUDA graph enabled and disabled. Graph-on is the primary OSCAR INT2 path; graph-off is kept for isolation and debugging.

`Peak (MiB)` is whole-GPU peak memory from `nvidia-smi`. `KV pool K+V (MiB, measured)` is only the K/V allocator slice parsed from the SGLang `KV Cache is allocated...` log line, not a theoretical estimate. Raw outputs are under `results/cuda_graph_compare_matrix/verify_readme_true_off_20260609/{off,on}/{short,medium,long,16k,32k}/`.

To rerun and summarize this matrix:

```bash
./scripts/cuda_graph_compare_matrix.sh \
  --tag "$(date +%Y%m%d)" \
  --presets short,medium,long,16k,32k \
  --modes bf16,int2,oscar-int2

python scripts/summarize_cuda_graph_matrix.py \
  results/cuda_graph_compare_matrix/<TAG>
```

### CUDA graph on

#### Prefill 512 tokens (~`short`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 382 | 4.98 | 76.98 | 82.38 | 3802 | 123 MiB |
| plain INT2 | 293<br>(-23%) | 4.72<br>(-5%) | 56.55<br>(-27%) | 64.28<br>(-22%) | 3799<br>(0%)<br>(-3 MiB) | 20 MiB<br>(-84%)<br>(-103 MiB) |
| OSCAR INT2 | 237<br>(-38%) | 5.81<br>(+17%) | 53.53<br>(-30%) | 60.72<br>(-26%) | 3839<br>(+1%)<br>(+37 MiB) | 102 MiB<br>(-17%)<br>(-21 MiB) |

#### Prefill 2048 tokens (~`medium`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 988 | 4.65 | 71.69 | 77.94 | 3875 | 246 MiB |
| plain INT2 | 1586<br>(+60%) | 4.51<br>(-3%) | 67.43<br>(-6%) | 73.92<br>(-5%) | 3836<br>(-1%)<br>(-39 MiB) | 41 MiB<br>(-83%)<br>(-205 MiB) |
| OSCAR INT2 | 813<br>(-18%) | 4.45<br>(-4%) | 52.03<br>(-27%) | 59.29<br>(-24%) | 4000<br>(+3%)<br>(+125 MiB) | 123 MiB<br>(-50%)<br>(-123 MiB) |

#### Prefill 8192 tokens (~`long`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 3121 | 3.88 | 64.64 | 67.19 | 4392 | 737 MiB |
| plain INT2 | 2409<br>(-23%) | 3.51<br>(-10%) | 52.31<br>(-19%) | 59.52<br>(-11%) | 3829<br>(-13%)<br>(-563 MiB) | 82 MiB<br>(-89%)<br>(-655 MiB) |
| OSCAR INT2 | 2579<br>(-17%) | 4.18<br>(+8%) | 46.76<br>(-28%) | 56.3<br>(-16%) | 4009<br>(-9%)<br>(-383 MiB) | 184 MiB<br>(-75%)<br>(-553 MiB) |

#### Prefill 16384 tokens (~`16k`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 3039 | 2.94 | 52.84 | 57.68 | 5007 | 1372 MiB |
| plain INT2 | 2795<br>(-8%) | 3<br>(+2%) | 53.57<br>(+1%) | 64.5<br>(+12%) | 4323<br>(-14%)<br>(-684 MiB) | 164 MiB<br>(-88%)<br>(-1208 MiB) |
| OSCAR INT2 | 2037<br>(-33%) | 2.59<br>(-12%) | 40.71<br>(-23%) | 52.55<br>(-9%) | 4419<br>(-12%)<br>(-588 MiB) | 287 MiB<br>(-79%)<br>(-1085 MiB) |

#### Prefill 32768 tokens (~`32k`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 1676 | 1.58 | 34.56 | 42.69 | 6607 | 2990 MiB |
| plain INT2 | 1815<br>(+8%) | 1.71<br>(+8%) | 42.27<br>(+22%) | 58.85<br>(+38%) | 5921<br>(-10%)<br>(-686 MiB) | 369 MiB<br>(-88%)<br>(-2621 MiB) |
| OSCAR INT2 | 2066<br>(+23%) | 1.78<br>(+13%) | 39.27<br>(+14%) | 51.09<br>(+20%) | 5937<br>(-10%)<br>(-670 MiB) | 512 MiB<br>(-83%)<br>(-2478 MiB) |

### CUDA graph off

#### Prefill 512 tokens (~`short`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 230 | 5.52 | 46.02 | 49.11 | 3806 | 123 MiB |
| plain INT2 | 175<br>(-24%) | 4.85<br>(-12%) | 34.61<br>(-25%) | 37.6<br>(-23%) | 3806<br>(0%)<br>(0 MiB) | 20 MiB<br>(-84%)<br>(-103 MiB) |
| OSCAR INT2 | 119<br>(-48%) | 5.06<br>(-8%) | 27.07<br>(-41%) | 28.18<br>(-43%) | 3818<br>(0%)<br>(+12 MiB) | 102 MiB<br>(-17%)<br>(-21 MiB) |

#### Prefill 2048 tokens (~`medium`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 704 | 4.93 | 44.72 | 48.4 | 3824 | 246 MiB |
| plain INT2 | 756<br>(+7%) | 4.67<br>(-5%) | 33.76<br>(-25%) | 37.49<br>(-23%) | 3806<br>(0%)<br>(-18 MiB) | 41 MiB<br>(-83%)<br>(-205 MiB) |
| OSCAR INT2 | 381<br>(-46%) | 4.51<br>(-9%) | 27.16<br>(-39%) | 28.24<br>(-42%) | 3944<br>(+3%)<br>(+120 MiB) | 123 MiB<br>(-50%)<br>(-123 MiB) |

#### Prefill 8192 tokens (~`long`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 3053 | 4.05 | 45.15 | 48.64 | 4344 | 737 MiB |
| plain INT2 | 2466<br>(-19%) | 3.65<br>(-10%) | 32.63<br>(-28%) | 32.72<br>(-33%) | 3806<br>(-12%)<br>(-538 MiB) | 82 MiB<br>(-89%)<br>(-655 MiB) |
| OSCAR INT2 | 2870<br>(-6%) | 3.55<br>(-12%) | 26.62<br>(-41%) | 27.82<br>(-43%) | 3978<br>(-8%)<br>(-366 MiB) | 184 MiB<br>(-75%)<br>(-553 MiB) |

- **Note (8K OSCAR, graph off):** the request layer aborts with `Connection reset by peer`; timing cells are **last-window partial metrics** before the crash.

#### Prefill 16384 tokens (~`16k`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 2874 | 3.01 | 40.12 | 43.41 | 4986 | 1372 MiB |
| plain INT2 | 1985<br>(-31%) | 2.64<br>(-12%) | 32.6<br>(-19%) | 38.5<br>(-11%) | 4302<br>(-14%)<br>(-684 MiB) | 164 MiB<br>(-88%)<br>(-1208 MiB) |
| OSCAR INT2 | 2154<br>(-25%) | 3.03<br>(+1%) | 24.13<br>(-40%) | 26.54<br>(-39%) | 4398<br>(-12%)<br>(-588 MiB) | 287 MiB<br>(-79%)<br>(-1085 MiB) |

#### Prefill 32768 tokens (~`32k`)

| Mode | Prefill<br>(tok/s) | Decode first<br>(tok/s) | Steady<br>(tok/s) | P95<br>(tok/s) | Peak<br>(MiB) | KV pool K+V<br>(MiB, measured) |
|------|----------------|------------------------|--------------|-----------|----------|---------------------|
| BF16 | 1657 | 1.62 | 32.9 | 39.8 | 6586 | 2990 MiB |
| plain INT2 | 2154<br>(+30%) | 1.96<br>(+21%) | 26.34<br>(-20%) | 36.14<br>(-9%) | 5900<br>(-10%)<br>(-686 MiB) | 369 MiB<br>(-88%)<br>(-2621 MiB) |
| OSCAR INT2 | 1883<br>(+14%) | 1.7<br>(+5%) | 21.77<br>(-34%) | 27.71<br>(-30%) | 5916<br>(-10%)<br>(-670 MiB) | 512 MiB<br>(-83%)<br>(-2478 MiB) |

### Derived: toggling CUDA graph (on vs off)

Steady decode **speedup** is graph-on steady ÷ graph-off steady. **Peak Δ** is graph-on peak MiB − graph-off peak MiB from whole-GPU `nvidia-smi` sampling.

| Prefill | BF16 steady × | INT2 steady × | OSCAR steady × | BF16 peak Δ | INT2 peak Δ | OSCAR peak Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.67× | 1.63× | 1.98× | -4 | -7 | +21 |
| 2K | 1.60× | 2.00× | 1.92× | +51 | +30 | +56 |
| 8K | 1.43× | 1.60× | 1.76× | +48 | +23 | +31 |
| 16K | 1.32× | 1.64× | 1.69× | +21 | +21 | +21 |
| 32K | 1.05× | 1.60× | 1.80× | +21 | +21 | +21 |

CUDA graph improves steady decode when the graph-off run completes. OSCAR **8K graph off** still hits the mixed-KV idle/crash path, so that row is a partial timing snapshot. Whole-GPU peak changes only modestly when graph is enabled (~+21 MiB at 16K/32K in this run).

Read the memory columns with two different lenses. **Peak (MiB)** is the whole-GPU `nvidia-smi` peak, including weights, CUDA context, workspaces, graph capture, KV, and other runtime memory. **KV pool K+V (MiB, measured)** is only the **K+V** allocator slice from the server log / bench CSV. Earlier memory-priority OSCAR 32K runs reached about 5775–5804 MiB total peak with smaller mixed-KV reserves, but those runs are not the same fair-cap configuration as the `--max-total-tokens 38272` matrix documented here.

Gate a 32K CSV against BF16 and plain INT2:

```bash
./scripts/regression_gate.sh --scenario balanced results/granite_bench_matrix/*/32k/bench_*.csv
```

Scenarios: `memory`, `balanced` (default), and `speed`.

To search for the largest stable prefill length on the local GPU:

```bash
python scripts/probe_max_prefill.py \
  --modes bf16,int2,oscar-int2 \
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

Install optional eval dependencies:

```bash
bash scripts/setup_eval_suite.sh
```

Run the full Granite suite across BF16, plain INT2, and OSCAR INT2:

```bash
bash rotation/granite-4.0-1b/eval_accuracy_suite.sh \
  --gpqa-num-examples 198 \
  --humaneval-num-examples 164 \
  --math-num-examples 500 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations
```

Pass `0` to skip an expensive benchmark, for example `--math-num-examples 0`. Each eval script also has `--help`.

To fill the remaining Granite accuracy gaps without rerunning completed tasks:

```bash
./scripts/run_granite_accuracy_fill.sh --only all
```

Baseline summary from `benchmarks/granite_accuracy_baseline.json`:

| Dataset | N target | N run | BF16 | plain INT2 | OSCAR INT2 | Status |
|---|---:|---:|---:|---:|---:|---|
| GPQA | 198 | 198 | 23.2% | 15.7% | 24.2% | complete |
| HumanEval | 164 | 164 / 164 / 61 | **31.8%** pass@1 | 0.0% | failed @ 61 | partial |
| MATH-500 | 500 | 20 (smoke) | 5.0% | 0.0% | 0.0% | smoke only |
| LCB v6 | full | 0 | — | — | — | not run |
| AIME 25 | full | 0 | — | — | — | not run |

On GPQA and HumanEval (BF16), OSCAR INT2 tracks or beats BF16 while plain INT2 collapses. HumanEval OSCAR INT2 needs a rerun after the mixed-KV server crash recorded in `benchmarks/RUN_LOG.md`.

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

```bash
./scripts/test.sh                              # repo unit tests + lightweight CLI smoke
./scripts/test.sh --unit-only                  # root tests/ unittest discovery only
./scripts/test.sh --run-integration            # optional SGLang dummy server

./scripts/probe.sh --model-path checkpoints/granite-4.0-1b-base --try-int2

# If FA3 fails on sm_120:
./scripts/probe.sh --model-path checkpoints/granite-4.0-1b-base --try-int2 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton
```

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

Useful test commands:

```bash
./scripts/test.sh
./scripts/test.sh --unit-only
./scripts/test.sh --skip-cli-smoke
./scripts/test.sh tests/test_bench_helpers.py
./scripts/test.sh tests.test_bench_helpers.BenchHelperTest.test_default_max_total_tokens_caps_single_request_long_context

# Optional environment-dependent checks:
./scripts/test.sh tests/test_gpu_cuda_optional.py
./scripts/test.sh --run-server-tests
./scripts/test.sh --run-integration
./scripts/test.sh --run-server-tests --test-port 31992
```

Acceptance-test examples: `probe.sh --try-int2` for a real SGLang server path, `bench.sh` / `bench_matrix.sh` for CUDA graph, decode speed, and VRAM peaks, and the Granite eval scripts for GPQA / HumanEval accuracy.

Test file index:

| Test file | Purpose |
|---|---|
| `tests/test_bench_helpers.py` | Unit tests for bench mode mapping, server command construction, OSCAR env vars, and KV estimate helpers. |
| `tests/test_cli_dry_run.py` | Dry-run coverage for `oscar_kv_quant.bench` report generation without starting SGLang. |
| `tests/test_gpu_cuda_optional.py` | Optional CUDA, BF16, SGLang, and FlashInfer environment checks. |
| `tests/test_kv_estimate.py` | Unit tests for BF16, INT2, and OSCAR mixed-KV byte estimates. |
| `tests/test_log_metrics.py` | Unit tests for parsing SGLang server logs into bench metrics. |
| `tests/test_longrun_gate.py` | Unit tests for long-run speed, memory, stability, and log-cleanliness gates. |
| `tests/test_probe_helpers.py` | Unit tests for probe log, health-check, status, and model-name helpers. |
| `tests/test_profiles.py` | Unit tests for model profile and KV geometry parsing. |
| `tests/test_regression_gate.py` | Unit tests for bench CSV regression gate scenarios. |
| `tests/test_rotation_gate.py` | Unit tests for rotation metadata and accuracy-drop gates. |
| `tests/test_sglang_dump_writer.py` | Optional test for the vendored SGLang async Q/K/V dump writer. |
| `tests/test_sglang_server_optional.py` | Optional dummy SGLang server probe test. |

No root-level test files are currently deprecated. Environment-dependent tests stay in the suite because they protect the local CUDA/SGLang validation path and skip cleanly when prerequisites are missing.

Common 32K pitfall: uncapped INT2 KV pools can balloon to 245632 tokens and look worse than BF16. Use the bench wrapper or pass `--max-total-tokens 38272`.

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
| `scripts/cuda_graph_compare_matrix.sh` | CUDA graph on/off matrix for BF16, INT2, and OSCAR INT2 |
| `scripts/summarize_cuda_graph_matrix.py` | Summarize CUDA graph matrix CSVs into README-style tables |
| `scripts/regression_gate.sh` / `oscar-kv-regression-gate` | CSV gate vs BF16/INT2 |
| `scripts/config_sweep.sh` / `oscar-kv-config-sweep` | Wrapper for configuration sweeps |
| `scripts/setup_eval_suite.sh` | Install optional accuracy-eval dependencies |
| `scripts/run_granite_accuracy_fill.sh` | Run remaining Granite accuracy fill jobs |
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
- CUDA graph support for `oscar-int8` and `oscar-int4` is not implemented yet. A proper fix likely needs a graph-safe quantized KV write path, such as Triton/CUDA kernels that compute/store scales on device or a separate precomputed-scale design; simply removing `.item()` would risk incorrect dequantization or stale per-layer scales.
- The official five-benchmark, five-repeat protocol is expensive on a laptop GPU; scale `eval_accuracy_suite.sh` incrementally.
