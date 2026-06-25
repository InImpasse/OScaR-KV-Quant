# OSCAR INT2 Final Convergence

## Goal

Final validation for `oscar_int2 = BF16 HP + INT2 bulk` in the llama.cpp / ggml CUDA path. The last remaining structural candidate was delayed LP materialization: keep tokens in the BF16 HP recent cache first, and write OSCAR2 INT2 LP bulk only for cold tokens.

## Candidate Tested

- Env-gated prototype: `LLAMA_KV_OSCAR2_DELAYED_LP=1`.
- Scope: CUDA FA, Granite D=128, `GGML_TYPE_OSCAR2_KV / GGML_TYPE_OSCAR2_KV`, HP prefill attention.
- Implementation shape tested:
  - Build cold-only LP index tensors from `slot_info.hp_batch_idxs`.
  - Skip LP set_rows for current-batch HP rows.
  - Write only non-HP cold rows to LP cache via `ggml_get_rows + ggml_set_rows`.
  - Keep HP writes and HP+LP attention unchanged.

The prototype built and ran after fixing cold-row shape back to `[head_dim, n_head, n_cold]`, but it did not pass the 8K speed gate. Runtime hot-path code was removed after the failed gate; no `LLAMA_KV_OSCAR2_DELAYED_LP` source-path hook should remain.

## Results

Model: `checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf`

Common env for `oscar_int2`:

```bash
LLAMA_KV_HP_SINK=64
LLAMA_KV_HP_RECENT=256
LLAMA_KV_HP_PREFILL_ATTENTION=1
LLAMA_KV_MIXED_VEC_RAW=1
LLAMA_KV_MIXED_VEC_MAIN=1
LLAMA_KV_HP_ALLOC_PAD=256
LLAMA_KV_HP_VIEW_TIGHT=1
```

| case | prompt | pp tok/s | result |
|---|---:|---:|---|
| oscar_int2 current | 512 | 1220.11 | smoke ok |
| oscar_int2 delayed LP prototype | 512 | 1355.46 | smoke ok |
| oscar_int2 delayed LP prototype | 2048 | 1310.53 | ok, no breakthrough |
| oscar_int2 delayed LP prototype | 8192 | 588.39 | failed Gate B |
| baseline_bf16 | 8192 | 3529.06 | 80% gate = 2823 tok/s |

Gate B required 8K prefill at >= 80% of BF16 with lower peak memory. The delayed-LP prototype reached only about 17% of BF16 speed, so quality and 16K/32K gates were not run.

## Do Not Repeat

- Do not re-enable `LLAMA_KV_OSCAR2_DELAYED_LP`; the hot-path prototype was removed after Gate B failure.
- Do not use `LLAMA_KV_HP_SKIP_LP_STORE_DIAG` as a final solution; earlier attempts showed it breaks graph/input assumptions or does not recover enough speed.
- Do not repeat HP window/mask/CUDA graph/V helper/K-affine sidecar/LP sampling/q2 scalar/old tiled kernel micro-tuning for `oscar_int2`.
- Do not run 16K/32K `oscar_int2` unless a new structural candidate first passes the 8K gate.

## Final Decision

Pure `oscar_int2` is frozen for now. The remaining deliverable route is `oscar_int4`, which has already shown healthy 32K memory reduction and speed close to or above BF16 in prior runs.
