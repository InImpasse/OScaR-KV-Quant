"""Theoretical KV cache size estimates (K+V only, no allocator overhead)."""

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
