# OSCAR Environment Requirements

This project validates **OSCAR (#1)** from `FutureMLS-Lab/OSCAR`, paper `arXiv:2605.17757`. The source of truth for runtime dependencies is the upstream `third_party/OSCAR/README.md` plus `third_party/OSCAR/sglang-research/python/pyproject.toml`.

## Upstream OSCAR Requirements

The upstream `FutureMLS-Lab/OSCAR` setup section requires:

- GPU: official 4B/8B runs use **1 x H100 80GB**; 32B / MiniMax / GLM runs require more H100 GPUs.
- CUDA: **CUDA 12.8 or 12.9**, with `nvcc` on `PATH`.
- Python: **Python 3.12**.
- Environment shape: **one environment** for both QKV dump and evaluation.
- Model weights: Hugging Face access for the target model weights.

The upstream README also notes that FlashInfer JIT compilation can fail if `nvcc` and the PyTorch CUDA build do not match. Set `CUDA_HOME` to the matching `cuda-12.x` toolkit before launching SGLang.

## Key SGLang Dependency Pins

The bundled `third_party/OSCAR/sglang-research/python/pyproject.toml` currently pins or constrains these important packages:

| Package | Version |
|------|------|
| `torch` | `2.9.1` |
| `transformers` | `5.3.0` |
| `cuda-python` | `12.9` |
| `flashinfer_python` | `0.6.7.post3` |
| `flashinfer_cubin` | `0.6.7.post3` |
| `flash-attn-4` | `>=4.0.0b4` |
| `sglang-kernel` | `0.4.1` |
| `kernels` | `0.13.0` |
| `torchao` | `0.9.0` |
| `torchaudio` | `2.9.1` |

These dependencies are resolved when installing `third_party/OSCAR/sglang-research/python`. Upstream currently leaves `kernels` unpinned, but this project pins `kernels==0.13.0` because newer `kernels` releases require `LayerRepository(revision=...|version=...)` and break `transformers==5.3.0` during SGLang startup.

## RTX 5050 / WSL Notes

RTX 5050 Laptop is a Blackwell / `sm_120` GPU. Upstream OSCAR targets H100-class systems, so RTX 5050 runs should be treated as best-effort smoke tests and local benchmarks:

- `flashinfer`, `sglang-kernel`, `flash-attn-4`, and Triton kernels must support `sm_120`.
- If `int2` or `fa3` fails, first try `--prefill-attention-backend triton --decode-attention-backend triton`.
- If JIT compilation reports a CUDA mismatch, compare `python -c "import torch; print(torch.version.cuda)"` with `nvcc --version`.

## Recommended Setup With uv

Run from the repository root:

```bash
chmod +x scripts/setup_env_uv.sh
./scripts/setup_env_uv.sh
```

The setup script explicitly runs `uv python install 3.12` before creating `.venv-oscar-kv`, so the project environment does not depend on the system default `python3`. Do not run the setup script with `sudo`. Existing `.venv-oscar-kv` environments are reused by default and are not replaced.

Useful environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export VENV_DIR=.venv-oscar-kv
export PYTHON_VERSION=3.12
export INSTALL_SGLANG=1
export DOWNLOAD_MODELS=0
export RUN_PROBE=1
./scripts/setup_env_uv.sh
```

Upstream OSCAR's vendored SGLang pins packages such as `torch==2.9.1`. This repository does not loosen those pins because PyTorch, FlashInfer, FlashAttention, SGLang kernels, and CUDA extensions are not guaranteed to be ABI-compatible across arbitrary newer versions. To avoid repeated long installs, the setup script skips reinstalling SGLang only when the key installed versions match the upstream pins. If any required package is missing or has a mismatched version, it installs the vendored stack.

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation:

```bash
./scripts/probe.sh
./scripts/probe.sh --try-dummy-server
./scripts/bench.sh --profile granite --preset short --modes bf16,int2 --dry-run
```

Download model checkpoints:

```bash
export HF_TOKEN=...
./scripts/download_models.sh
```

## Manual Fallback

```bash
python3.12 -m venv .venv-oscar-kv
source .venv-oscar-kv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install -e third_party/OSCAR/sglang-research/python
oscar-kv-probe
```

## Configuration Record

This repository also records the upstream requirements and key dependency pins in `pyproject.toml` under `[tool.oscar_kv_quant.environment]` for future scripts or CI checks.
