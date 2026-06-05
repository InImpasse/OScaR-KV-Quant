# GitHub Issue: FP4 KV Cache Decode

## Title

FP4 KV cache decode path fails because MHATokenToKVPoolFP4.set_kv_buffer does not accept is_decode

## Body

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
