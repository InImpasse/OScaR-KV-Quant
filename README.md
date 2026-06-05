# OSCAR-KV-Quant

Validation harness for **OSCAR (#1)**, [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR), paper [arXiv:2605.17757](https://arxiv.org/abs/2605.17757), focused on KV-cache memory and decode throughput experiments on WSL + NVIDIA RTX 5050 8GB.

This repository is **not** OScaR (#2), `ZunhaiSu/OScaR-KV-Quant`.

## What This Repo Does

- Installs and wraps the official OSCAR code as `third_party/OSCAR`.
- Provides local validation scripts for Granite 4.0 1B and Gemma 4 E2B checkpoints.
- Measures BF16 / FP8 / FP4 / INT2 baseline KV-cache modes exposed by SGLang.
- Measures OSCAR INT2 mode when rotation files are available.
- Writes CSV and Markdown reports with throughput, GPU memory, log paths, and theoretical KV estimates.

## Requirements

Upstream OSCAR and the bundled SGLang tree are strict about the runtime stack:

- Python 3.12 for the project environment.
- CUDA 12.8 or 12.9 with `nvcc` on `PATH`.
- Working NVIDIA driver / WSL GPU passthrough (`nvidia-smi` should work).
- `ninja` for SGLang / TVM FFI JIT kernel builds. The setup script installs the Python wheel.
- `uv` is recommended for environment creation.
- Hugging Face access for gated checkpoints when downloading models.

Run the host check before installation:

```bash
./scripts/check_env.sh
```

If the check reports an older default `python3`, that is fine as long as `python3.12` is available through `uv` or the system package manager. If the CUDA toolkit is not 12.8 or 12.9, install a matching toolkit and export `CUDA_HOME`.

## Step-by-Step Validation Flow

### 1. Enter the Repository

```bash
cd OSCAR-KV-Quant
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Create the Python Environment

Use the uv-based setup script. It explicitly runs `uv python install 3.12`, creates `.venv-oscar-kv` with Python 3.12, installs this package, installs OSCAR's vendored SGLang, and runs a basic probe.

```bash
# Do not run this script with sudo.
./scripts/setup_env_uv.sh
```

The rest of this README uses wrapper scripts under `scripts/`, so you do not need to manually activate `.venv-oscar-kv`.

If you want an interactive shell inside the environment, activation is still available:

```bash
source .venv-oscar-kv/bin/activate
hash -r
```

The setup script reuses an existing `.venv-oscar-kv` by default. It does not delete it and does not ask to replace it.

If wrapper scripts report missing commands, rerun setup:

```bash
./scripts/setup_env_uv.sh
```

The vendored SGLang stack is pinned by upstream OSCAR. The setup script skips reinstalling SGLang only when the key package versions in `.venv-oscar-kv` match the upstream pins, including `torch==2.9.1`, `transformers==5.3.0`, `kernels==0.13.0`, `flashinfer_python==0.6.7.post3`, and `sglang-kernel==0.4.1`. If versions are missing or mismatched, it installs the vendored stack. The `kernels` pin is added locally because newer `kernels` releases break `transformers==5.3.0` during SGLang startup.

Run the lightweight unit tests before GPU work:

```bash
./scripts/test.sh
```

The default test suite includes pure-Python tests plus optional CUDA/import tests. Optional tests are skipped automatically when `torch`, CUDA, `sglang`, or `flashinfer` are unavailable.

To include the optional SGLang dummy server test after the environment is installed:

```bash
OSCAR_KV_RUN_SERVER_TESTS=1 ./scripts/test.sh
```

The server test is intentionally opt-in because it starts a local SGLang process.

If you install CUDA 12.8 or 12.9 later, export the matching path before setup or before running probes:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### 4. Check or Download Models

The download script skips a model only when both `config.json` and at least one model weight file are present. If a previous download left only configs or tokenizer files, rerun the script to resume the missing weights.

```bash
ls checkpoints/granite-4.0-1b-base/config.json
ls checkpoints/gemma-4-E2B/config.json

# Only needed if a checkpoint is missing.
export HF_TOKEN=...  # required for gated models after accepting terms
./scripts/download_models.sh
```

### 5. Probe the Environment

Start with the lightweight probe:

```bash
./scripts/probe.sh
```

Expected good signs:

- `torch.cuda.is_available == True`
- GPU name is the RTX 5050
- `sglang_import_ok: true`
- `flashinfer_import_ok: true`

The dummy SGLang server path is optional. In the current upstream tree, `--model-path dummy` can still enter Hugging Face config loading, so the first meaningful server check is the local Granite BF16 probe in the next step.

### 6. Probe Granite Baseline Paths

Run BF16 first:

```bash
./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --kv-cache-dtype bf16 \
  --timeout-s 180
```

Then test the INT2 path:

```bash
./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --try-int2 \
  --timeout-s 180
```

If INT2 fails with FA3-related issues, force Triton for both prefill and decode:

```bash
./scripts/probe.sh \
  --model-path checkpoints/granite-4.0-1b-base \
  --try-int2 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --timeout-s 180
```

### 7. Run Granite Baseline Benchmarks

Start with short context:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes bf16,fp8,fp4,int2 \
  --request-api completions \
  --results-dir results/granite_short
```

Then try medium context:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset medium \
  --modes bf16,fp8,fp4,int2 \
  --request-api completions \
  --results-dir results/granite_medium
```

Only try long context after short and medium succeed:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset long \
  --modes bf16,fp8,fp4,int2 \
  --request-api completions \
  --results-dir results/granite_long
```

Mode meanings:

- `bf16`: BF16 KV-cache baseline.
- `fp8`: SGLang `fp8_e4m3` KV-cache comparison mode.
- `fp4`: SGLang `fp4_e2m1` KV-cache comparison mode.
- `int2`: SGLang INT2 KV-cache baseline without OSCAR rotations.

### 8. Generate Granite OSCAR Rotations

Use a small calibration budget first on RTX 5050:

```bash
DUMP_KVCACHE_TOKENS=2000 NUM_WORKERS=2 \
bash rotation/granite-4.0-1b/save_qkv_granite.sh
```

Then compute rotations:

```bash
bash rotation/granite-4.0-1b/compute_rotation.sh
```

Find the latest rotation directory:

```bash
ROT_DIR=$(ls -1dt rotation/granite-4.0-1b/GPQA/seq*_prompt*_group*/rotations | head -1)
echo "$ROT_DIR"
```

### 9. Run Granite OSCAR INT2

First dry-run the command and environment:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes oscar-int2 \
  --rot-dir "$ROT_DIR" \
  --request-api completions \
  --dry-run \
  --results-dir results/granite_oscar_dry_run
```

Then run the real short benchmark:

```bash
./scripts/bench.sh \
  --profile granite \
  --preset short \
  --modes oscar-int2 \
  --rot-dir "$ROT_DIR" \
  --request-api completions \
  --results-dir results/granite_oscar_short
```

### 10. Probe and Benchmark Gemma4

Start with probes:

```bash
./scripts/probe.sh \
  --model-path checkpoints/gemma-4-E2B \
  --kv-cache-dtype bf16 \
  --timeout-s 180

./scripts/probe.sh \
  --model-path checkpoints/gemma-4-E2B \
  --try-int2 \
  --timeout-s 180
```

Then run only short context first:

```bash
./scripts/bench.sh \
  --profile gemma4 \
  --preset short \
  --modes bf16,fp8,fp4,int2 \
  --request-api completions \
  --results-dir results/gemma4_short
```

Gemma4 may exceed 8GB VRAM more easily than Granite. Do not move to medium or long until the short run succeeds.

### 11. Inspect Results

Each benchmark writes CSV and Markdown reports:

```bash
ls results/**/*.csv
ls results/**/*.md
```

Key fields:

- `server_ok`: whether the run completed.
- `error`: failure reason or log tail summary.
- `decode_toks_per_sec`: decode throughput.
- `baseline_mib`: GPU memory before server start.
- `peak_mib_total`: peak whole-GPU memory during the run.
- `peak_mib_delta`: `peak_mib_total - baseline_mib`.
- `kv_theory_bf16_gib`: theoretical BF16 KV size.
- `kv_theory_selected_gib`: theoretical KV size for the selected mode.

When interpreting results, compare both theoretical KV memory and observed total GPU memory. On 8GB GPUs, total memory may be dominated by weights, workspaces, kernel caches, and allocator reserve rather than KV cache alone.

## Documentation

- Environment setup: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- RTX 5050 reproduction flow: [docs/repro_5050.md](docs/repro_5050.md)
- Model and KV dtype support notes: [docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md)
- Rotation scripts: [rotation/README.md](rotation/README.md)

## Important Limitations

- Official OSCAR results target H100-class GPUs. RTX 5050 / Blackwell `sm_120` support depends on SGLang, FlashInfer, FlashAttention, Triton, and kernel compatibility.
- SGLang does **not** expose native `int8` or `int4` KV-cache dtypes in this codebase. This repo maps `int8` to `fp8_e4m3` and `int4` to `fp4_e2m1` as comparison modes.
- Full paper-style accuracy runs such as GPQA, HumanEval, LiveCodeBench, AIME, and MATH-500 are not part of the first RTX 5050 smoke path.

## Upstream Relationship

- `third_party/OSCAR` is the official OSCAR repository as a git submodule.
- Upstream calibration and evaluation templates live under `third_party/OSCAR/rotation/`.
- This repo adds Granite and Gemma4 adaptation scripts under `rotation/`.

## License

This harness should be treated as MIT-style project glue unless a file states otherwise. `third_party/OSCAR` follows its upstream license.
