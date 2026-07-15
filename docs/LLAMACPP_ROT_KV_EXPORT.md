# llama.cpp Rotated KV GGUF Export

This repo's llama.cpp branch consumes OSCAR K/V rotations as optional GGUF
tensors:

- `blk.N.attn_k_rot.weight`
- `blk.N.attn_v_rot.weight`

The runtime does not need to read PyTorch `.pt` files. Use the `.pt` rotation
checkpoints as an intermediate artifact, then bake them into a normal GGUF.

## End-To-End From A Dataset

If no `.pt` rotation checkpoints exist yet, provide a calibration dataset first.
The dataset can be either a plain text file with one prompt per line or JSONL
with one of `prompt`, `text`, `question`, or `input` fields.

This repo can export the same GPQA/GSM8K prompt format used by the local eval
harness:

```bash
python3 scripts/export_calibration_prompts.py \
  --out data/calibration_prompts_gpqa_gsm8k.jsonl \
  --datasets gpqa,gsm8k \
  --gpqa-n-cases 198 \
  --gsm8k-n-cases 100 \
  --seed 1234 \
  --shuffle
```

Start with a small smoke run:

```bash
python3 scripts/build_llamacpp_rot_kv_gguf.py \
  --base checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
  --dataset data/calibration_prompts_gpqa_gsm8k.jsonl \
  --work-dir runs/llamacpp_rot_kv_calibration_smoke \
  --out checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
  --max-prompts 8 \
  --dump-token-budget 2000 \
  --ctx 4096 \
  --dry-run
```

Run it for real only after checking the command:

```bash
python3 scripts/build_llamacpp_rot_kv_gguf.py \
  --base checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
  --dataset data/calibration_prompts_gpqa_gsm8k.jsonl \
  --work-dir runs/llamacpp_rot_kv_calibration_current \
  --out checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
  --max-prompts 198 \
  --dump-token-budget 30000 \
  --ctx 4096 \
  --overwrite \
  --ack-run
```

This runs:

1. `scripts/dump_llamacpp_qkv_for_rotation.py`
2. `scripts/compute_llamacpp_kv_rotation.py`
3. `scripts/export_llamacpp_rot_kv_gguf.py`

The intermediate rotation files are written under:

```text
runs/llamacpp_rot_kv_calibration_current/rotations/
  k_rotation_qqt_r_h_pbr.pt
  v_rotation_sst_r_h_pbr.pt
  rotation_meta.json
```

## Dump Format

The llama.cpp calibration dump now matches the quantization-tool llama.cpp path:

- Q/K/V `.pt` tensors are stored as `bfloat16`
- `seq_lens/*.pt` are stored as `int32`
- `calibration_meta.json` records `dump_dtype: "bfloat16"`
- OSKV streaming dump, token budget, partial resume, and raw dump cleanup are supported

For Granite paper-style calibration, use roughly 30K tokens. A 30K bf16 dump is
typically about 7 GB on disk, roughly half the old float32 layout.

## Resume

`build_llamacpp_rot_kv_gguf.py --resume` reuses an existing dump or rotation
stage when the stage manifest under `work-dir/stage_manifests/` still matches the
requested inputs. You can also pass `--resume-partial` to
`dump_llamacpp_qkv_for_rotation.py` to continue an interrupted dump.

## Export

```bash
python3 scripts/export_llamacpp_rot_kv_gguf.py \
  --base checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
  --rot-dir runs/llamacpp_rot_kv_calibration_current/rotations \
  --out checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf
```

If `--out` is omitted, the script writes next to `--base` using the
`-rot-kv.gguf` suffix. The base weights are copied unchanged; only the rotation
tensors are appended. Existing rotation tensors are rejected unless
`--replace-rotations` is passed.

## Validation

After export, verify that the baked tensors match the `.pt` rotations:

```bash
python3 scripts/check_granite_rotation_alignment.py \
  --rot-dir runs/llamacpp_rot_kv_calibration_current/rotations \
  --gguf checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf
```

## Calibration Data

You do not need a validation dataset to bake an existing rotation into GGUF.
You only need:

- a base GGUF
- `k_rotation_qqt_r_h_pbr.pt`
- `v_rotation_sst_r_h_pbr.pt`

You do need calibration data if you want to create new `.pt` rotation files for
a model. The llama.cpp flow is:

1. Run model prompts and dump post-RoPE Q/K/V activations.
2. Fit per-layer K/V rotations from the QKV dump.
3. Validate `rotation_meta.json` for calibration size and orthogonality.
4. Bake the `.pt` rotations into a `*-rot-kv.gguf` for normal `llama-cli` /
   `llama-bench` usage.

The official Granite recipe used GPQA and roughly 30K calibration tokens. The
llama.cpp scripts do not enforce a specific dataset; use prompts that resemble
the target workload, then validate quality separately against BF16.
