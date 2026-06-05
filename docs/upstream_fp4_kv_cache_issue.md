# FP4 KV Cache Decode Issue

This document contains a clean upstream-facing bug report and PR description for the FP4 KV cache decode issue.

## Short Summary

The FP4 KV cache decode path fails because the shared Triton attention decode caller passes `is_decode=True` to `set_kv_buffer()`, while `MHATokenToKVPoolFP4.set_kv_buffer()` does not accept that keyword argument.

The fix is to make the FP4 override accept the same optional keyword as the base MHA KV pool:

```python
is_decode: bool = False
```

No behavior change is required; this is an interface compatibility fix.

## Failure

### Reproduction Command

```bash
python -m sglang.launch_server \
  --model-path <model-path> \
  --kv-cache-dtype fp4_e2m1 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --trust-remote-code
```

### Observed Error

```text
TypeError: MHATokenToKVPoolFP4.set_kv_buffer() got an unexpected keyword argument 'is_decode'
```

### Relevant Call Site

The shared Triton decode path calls:

```python
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer,
    forward_batch.out_cache_loc,
    k,
    v,
    layer.k_scale,
    layer.v_scale,
    is_decode=True,
)
```

The base MHA KV pool accepts `is_decode`, but the FP4 override does not.

## Proposed Fix

Update `MHATokenToKVPoolFP4.set_kv_buffer()` to accept `is_decode`:

```python
def set_kv_buffer(
    self,
    layer: RadixAttention,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    layer_id_override: Optional[int] = None,
    is_decode: bool = False,
):
```

The parameter can remain unused for now. This keeps the FP4 implementation compatible with the shared Triton decode caller.

## Validation

The same FP4 launch configuration was tested against two branches:

| Branch | Result |
| --- | --- |
| `main` | Fails with `TypeError: MHATokenToKVPoolFP4.set_kv_buffer() got an unexpected keyword argument 'is_decode'` |
| `bugfix/fp4-kv-cache-decode-write` | Succeeds; server reaches healthy state |

As a control, the FP8 KV cache path was also tested on both branches with the same launch shape except `--kv-cache-dtype fp8_e4m3`:

| Branch | FP8 Result |
| --- | --- |
| `main` | Succeeds; server reaches healthy state |
| `bugfix/fp4-kv-cache-decode-write` | Succeeds; server reaches healthy state |

This suggests the change is limited to FP4 decode-path compatibility and does not affect the existing FP8 path.

Successful run indicators from the bugfix branch:

```text
Using KV cache dtype: torch.float4_e2m1fn_x2
KV Cache is allocated.
GET /health HTTP/1.1" 200 OK
model_server_ok: true
```

## Git Branch And Commit

Branch:

```text
bugfix/fp4-kv-cache-decode-write
```

Commit:

```text
26b329a Fix FP4 KV cache decode writes
```

Commit author:

```text
InImpasse <tonyjiang@impasse.top>
```

The commit contains no co-author trailer.

## GitHub Issue Draft

### Title

```text
FP4 KV cache decode path fails because MHATokenToKVPoolFP4.set_kv_buffer does not accept is_decode
```

### Body

```markdown
## Summary

The FP4 KV cache decode path can fail because the shared Triton attention decode caller passes `is_decode=True` to `set_kv_buffer()`, but `MHATokenToKVPoolFP4.set_kv_buffer()` does not accept that keyword argument.

The base MHA KV pool accepts `is_decode`, but the FP4 override does not.

## Reproduction

Run a causal LM server with:

```bash
python -m sglang.launch_server \
  --model-path <model-path> \
  --kv-cache-dtype fp4_e2m1 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --trust-remote-code
```

## Observed Error

```text
TypeError: MHATokenToKVPoolFP4.set_kv_buffer() got an unexpected keyword argument 'is_decode'
```

The caller passes:

```python
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer,
    forward_batch.out_cache_loc,
    k,
    v,
    layer.k_scale,
    layer.v_scale,
    is_decode=True,
)
```

## Expected Behavior

`MHATokenToKVPoolFP4.set_kv_buffer()` should accept the same optional decode-path keyword as the base MHA KV pool.

## Proposed Fix

Add a defaulted `is_decode` parameter to the FP4 method signature:

```python
def set_kv_buffer(
    self,
    layer: RadixAttention,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    layer_id_override: Optional[int] = None,
    is_decode: bool = False,
):
```

The parameter does not need to change FP4 behavior immediately; accepting it keeps the FP4 override compatible with the shared decode caller.

## Validation

The same FP4 server launch was tested on two branches:

- `main`: fails with `TypeError: MHATokenToKVPoolFP4.set_kv_buffer() got an unexpected keyword argument 'is_decode'`
- `bugfix/fp4-kv-cache-decode-write`: succeeds and reaches a healthy server state

As a control, the FP8 KV cache path succeeds on both `main` and `bugfix/fp4-kv-cache-decode-write`.
```

## Pull Request Draft

### Title

```text
Fix FP4 KV cache decode writes
```

### Body

```markdown
## Summary

Fixes FP4 KV cache decode startup by allowing `MHATokenToKVPoolFP4.set_kv_buffer()` to accept the `is_decode` keyword used by the shared Triton attention decode path.

The parameter is accepted for interface compatibility with the base MHA KV pool and shared decode caller. No FP4 behavior is changed.

## Validation

- `python3 -m py_compile sglang-research/python/sglang/srt/mem_cache/memory_pool.py`
- Compared the same FP4 launch configuration on:
  - `main`: reproduced `TypeError: MHATokenToKVPoolFP4.set_kv_buffer() got an unexpected keyword argument 'is_decode'`
  - `bugfix/fp4-kv-cache-decode-write`: server reached `/health` OK
- Compared the FP8 KV cache path as a control:
  - `main`: server reached `/health` OK
  - `bugfix/fp4-kv-cache-decode-write`: server reached `/health` OK
```

## Push And PR Commands

```bash
cd ~/project/OSCAR
git push -u origin bugfix/fp4-kv-cache-decode-write
```

Then open a PR from:

```text
InImpasse/OSCAR:bugfix/fp4-kv-cache-decode-write
```

to the upstream default branch.
