# RTX 5050 Reproduction Guide

This guide describes the RTX 5050 / WSL smoke and benchmark path for **OSCAR (#1)** from [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR), paper [arXiv:2605.17757](https://arxiv.org/abs/2605.17757). It is not for OScaR (#2).

## Hardware and Risk Notes

- **RTX 5050 Laptop is Blackwell / `sm_120`**. Upstream OSCAR targets H100 + CUDA 12.8/12.9.
- `flashinfer`, `sglang-kernel`, FlashAttention, or Triton may fail to compile or may not support the required runtime path.
- Always start with `oscar-kv-probe` before launching model benchmarks.

## Recommended Environment

- Ubuntu 22.04（WSL2）
- NVIDIA driver with CUDA 12.x support matching the PyTorch wheel.
- CUDA **12.8 or 12.9**, with `nvcc` on `PATH`.
- Python **3.12**.
- `uv` is recommended; Conda or venv can also work.

See [ENVIRONMENT.md](ENVIRONMENT.md) for full details. The bundled `sglang-research` currently pins `torch==2.9.1`, `transformers==5.3.0`, `flashinfer_python==0.6.7.post3`, `flash-attn-4>=4.0.0b4`, and `sglang-kernel==0.4.1`.

## One-Step Setup With uv

```bash
chmod +x scripts/setup_env_uv.sh
./scripts/setup_env_uv.sh
```

If CUDA needs to be pinned explicitly:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
./scripts/setup_env_uv.sh
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Submodules and SGLang

```bash
git submodule update --init --recursive
./scripts/install_sglang_os.sh
```

`install_sglang_os.sh` installs `third_party/OSCAR/sglang-research/python` in editable mode. If JIT compilation fails, set `CUDA_HOME` to the CUDA toolkit matching the installed PyTorch CUDA build.

## Model Checkpoints

`checkpoints/` is gitignored. Download the target models with:

```bash
export HF_TOKEN=...   # required for gated models after accepting terms on Hugging Face
./scripts/download_models.sh
```

Default outputs:

- `checkpoints/granite-4.0-1b-base`
- `checkpoints/gemma-4-e2b`

## Probe and Baseline Benchmarks

```bash
oscar-kv-probe
oscar-kv-probe --try-dummy-server
oscar-kv-probe --model-path checkpoints/granite-4.0-1b-base --kv-cache-dtype bf16
oscar-kv-probe --model-path checkpoints/granite-4.0-1b-base --try-int2

# Print the SGLang command, OSCAR environment, and theoretical KV estimate only.
oscar-kv-bench --profile granite --preset short --modes bf16,int2 --dry-run

# Use completions for base models; it avoids chat-template assumptions.
oscar-kv-bench --profile granite --preset short --modes bf16,int2 --request-api completions
```

See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for KV dtype mapping. This SGLang tree has no native INT8 / INT4 KV dtype; this project uses `fp8_e4m3` and `fp4_e2m1` as comparison modes.

Recommended RTX 5050 smoke order:

1. `oscar-kv-probe`
2. `oscar-kv-probe --try-dummy-server`
3. `oscar-kv-probe --model-path checkpoints/granite-4.0-1b-base --kv-cache-dtype bf16`
4. `oscar-kv-probe --model-path checkpoints/granite-4.0-1b-base --try-int2`
5. `oscar-kv-bench --profile granite --preset short --modes bf16,int2 --request-api completions`
6. After generating or downloading rotation files, run `--modes oscar-int2 --rot-dir ...`.

`oscar-kv-bench` reports `baseline_mib`, `peak_mib_total`, `peak_mib_delta`, `server_log_path`, and theoretical KV GiB. `peak_mib_delta` is a whole-GPU `nvidia-smi` delta from the pre-run baseline, not an exact PyTorch allocator or per-process memory value.

## OSCAR Rotation Calibration

The scripts under `rotation/granite-4.0-1b/` and `rotation/gemma-4-e2b/` are adapted from the upstream `rotation/qwen3-8B` templates. Full QKV dumps can be memory intensive; on RTX 5050 start with a small `DUMP_KVCACHE_TOKENS` value for smoke testing.
