"""KV cache size estimates used by bench summaries.

The helpers intentionally model the runtime layout closely enough to keep
memory gates honest. The OSCAR mixed-int2 path stores quantized middle tokens
in a packed int2 arena plus scale/zero metadata, while prefix/recent windows
live in a separate high-precision arena.
"""

from __future__ import annotations

# Effective bits per element for OSCAR INT2 + mixed BF16 windows (paper ~2.28 BPE at 128K).
OSCAR_INT2_EFFECTIVE_BITS_PER_ELEM = 2.28


def kv_bytes_bf16(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> float:
    """K and V in bf16: 2 tensors × fp16 size."""
    elems = 2 * num_layers * num_kv_heads * seq_len * head_dim
    return elems * 2


def kv_bytes_int2_packed_naive(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> float:
    """Pure INT2 storage (no mixed windows): 2 bits per weight → /4 from bf16."""
    return kv_bytes_bf16(num_layers, num_kv_heads, seq_len, head_dim) / 8.0


def kv_bytes_int2_runtime(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    v_head_dim: int | None = None,
    group_size: int | None = 128,
    scale_dtype_bytes: int = 2,
) -> float:
    """Runtime int2 KV bytes including per-group scale/zero metadata."""
    v_dim = head_dim if v_head_dim is None else v_head_dim
    g_k = head_dim if group_size is None else group_size
    g_v = v_dim if group_size is None else group_size
    if head_dim % 4 != 0 or v_dim % 4 != 0:
        raise ValueError("int2 KV estimate requires K/V head dims divisible by 4")
    if g_k <= 0 or g_v <= 0 or head_dim % g_k != 0 or v_dim % g_v != 0:
        raise ValueError("int2 KV estimate requires group_size dividing K/V dims")
    per_head_token = (
        head_dim // 4
        + v_dim // 4
        + 2 * scale_dtype_bytes * (head_dim // g_k + v_dim // g_v)
    )
    return num_layers * num_kv_heads * seq_len * per_head_token


def kv_bytes_oscar_mixed_runtime_estimate(
    num_layers: int,
    num_kv_heads: int,
    quant_tokens: int,
    head_dim: int,
    *,
    v_head_dim: int | None = None,
    prefix_bf16: int = 64,
    recent_bf16: int = 256,
    max_running_requests: int = 1,
    hp_prefix_pool_tokens: int | None = None,
    group_size: int | None = 128,
    scale_dtype_bytes: int = 2,
    hp_dtype_bytes: int = 2,
    page_size: int = 8,
) -> float:
    """Approximate actual OSCAR mixed-int2 runtime KV bytes.

    ``quant_tokens`` is the quant arena cap (usually SGLang max_total_tokens).
    HP-prefix pool follows the runtime default: at least 1024 slots, or
    ``max_running_requests * prefix_bf16`` when that is larger. The per-request
    recent ring reserves ``recent_bf16 + page_size - 1`` slots.
    """
    v_dim = head_dim if v_head_dim is None else v_head_dim
    quant_bytes = kv_bytes_int2_runtime(
        num_layers,
        num_kv_heads,
        quant_tokens,
        head_dim,
        v_head_dim=v_dim,
        group_size=group_size,
        scale_dtype_bytes=scale_dtype_bytes,
    )
    prefix_pool = (
        max(1024, max_running_requests * prefix_bf16)
        if hp_prefix_pool_tokens is None
        else hp_prefix_pool_tokens
    )
    prefix_pool = ((prefix_pool + page_size - 1) // page_size) * page_size
    hp_slots = prefix_pool + max_running_requests * (recent_bf16 + page_size - 1)
    hp_bytes = (
        num_layers
        * num_kv_heads
        * hp_slots
        * (head_dim + v_dim)
        * hp_dtype_bytes
    )
    return quant_bytes + hp_bytes


def kv_bytes_oscar_mixed_estimate(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    prefix_bf16: int = 64,
    recent_bf16: int = 256,
    effective_bits: float = OSCAR_INT2_EFFECTIVE_BITS_PER_ELEM,
) -> float:
    """Rough K+V bytes when middle is INT2-effective and ends are BF16 (both K and V)."""
    if seq_len <= prefix_bf16 + recent_bf16:
        return kv_bytes_bf16(num_layers, num_kv_heads, seq_len, head_dim)
    mid = seq_len - prefix_bf16 - recent_bf16
    bf16_tokens = prefix_bf16 + recent_bf16
    elems_bf16 = 2 * num_layers * num_kv_heads * bf16_tokens * head_dim
    elems_mid = 2 * num_layers * num_kv_heads * mid * head_dim
    bytes_bf16 = elems_bf16 * 2
    bytes_mid = elems_mid * (effective_bits / 8.0)
    return bytes_bf16 + bytes_mid


def fmt_gib(x: float) -> str:
    return f"{x / (1024**3):.4f}"
