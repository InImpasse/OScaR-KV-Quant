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
| `bugfix/fp4-kv-cache-decode-write` | Succeeds; server reaches `/health` OK |

As a control, the FP8 KV cache path was also tested on both branches with the same launch shape except `--kv-cache-dtype fp8_e4m3`:

| Branch | FP8 Result |
| --- | --- |
| `main` | Succeeds; server reaches `/health` OK |
| `bugfix/fp4-kv-cache-decode-write` | Succeeds; server reaches `/health` OK |

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
