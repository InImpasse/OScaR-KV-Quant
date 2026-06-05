# Rotation Calibration Scripts

These scripts are adapted from the upstream `third_party/OSCAR/rotation/qwen3-8B` templates. Run them from the OSCAR-KV-Quant repository root, with `third_party/OSCAR` checked out as a submodule.

## Granite 4.0 1B

```bash
cd /path/to/OSCAR-KV-Quant
bash rotation/granite-4.0-1b/save_qkv_granite.sh
# After a successful dump
bash rotation/granite-4.0-1b/compute_rotation.sh
```

On RTX 5050, start with smaller values such as `DUMP_KVCACHE_TOKENS=2000` and `NUM_WORKERS=2`.

## Gemma 4 E2B

```bash
bash rotation/gemma-4-e2b/save_qkv_gemma4.sh
bash rotation/gemma-4-e2b/compute_rotation.sh
```

`compute_rotation.sh` first tries to infer `NUM_LAYERS` and `HEAD_DIM` from the local checkpoint `config.json` or nested `text_config`. If the parsed values do not match the actual upstream OSCAR dump layout, override them explicitly:

```bash
HEAD_DIM=128 NUM_LAYERS=40 bash rotation/granite-4.0-1b/compute_rotation.sh
```

## Hadamard Baseline Without QKV Dump

```bash
METHOD=hadamard bash rotation/granite-4.0-1b/compute_rotation.sh
```

Then pass the generated rotation directory to:

```bash
oscar-kv-bench --profile granite --modes oscar-int2 --rot-dir /path/to/rotations
```

## Preflight Checks

The dump scripts check these conditions before starting a server:

- `MODEL/config.json` exists.
- `third_party/OSCAR/sglang-dump-qkv/python` exists.
- `curl` is available.
- The active Python can run `sglang.launch_server`.

If any check fails, install the environment and download checkpoints first. See `docs/ENVIRONMENT.md` and `docs/repro_5050.md`.
