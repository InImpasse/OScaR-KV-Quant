import unittest

from oscar_kv_quant.kv_estimate import (
    kv_bytes_bf16,
    kv_bytes_int2_packed_naive,
    kv_bytes_int2_runtime,
    kv_bytes_oscar_mixed_estimate,
    kv_bytes_oscar_mixed_runtime_estimate,
)


class KVEstimateTest(unittest.TestCase):
    def test_bf16_kv_bytes_counts_k_and_v(self) -> None:
        self.assertEqual(
            kv_bytes_bf16(num_layers=2, num_kv_heads=4, seq_len=8, head_dim=16),
            2 * 2 * 4 * 8 * 16 * 2,
        )

    def test_naive_int2_is_eighth_of_bf16_bytes(self) -> None:
        bf16 = kv_bytes_bf16(4, 2, 128, 64)
        self.assertEqual(kv_bytes_int2_packed_naive(4, 2, 128, 64), bf16 / 8)

    def test_runtime_int2_includes_scales_and_zeros(self) -> None:
        packed_only = kv_bytes_int2_packed_naive(4, 2, 128, 64)
        runtime = kv_bytes_int2_runtime(
            4, 2, 128, 64, group_size=32, scale_dtype_bytes=2
        )
        self.assertGreater(runtime, packed_only)

    def test_oscar_mixed_uses_bf16_when_sequence_inside_windows(self) -> None:
        self.assertEqual(
            kv_bytes_oscar_mixed_estimate(
                4, 2, seq_len=128, head_dim=64, prefix_bf16=64, recent_bf16=256
            ),
            kv_bytes_bf16(4, 2, 128, 64),
        )

    def test_oscar_mixed_is_smaller_than_bf16_for_long_sequences(self) -> None:
        mixed = kv_bytes_oscar_mixed_estimate(
            4, 2, seq_len=1024, head_dim=64, prefix_bf16=64, recent_bf16=256
        )
        self.assertLess(mixed, kv_bytes_bf16(4, 2, 1024, 64))

    def test_oscar_runtime_estimate_includes_hp_reserve(self) -> None:
        one_req = kv_bytes_oscar_mixed_runtime_estimate(
            4,
            2,
            quant_tokens=1024,
            head_dim=64,
            prefix_bf16=64,
            recent_bf16=256,
            max_running_requests=1,
            group_size=64,
        )
        two_req = kv_bytes_oscar_mixed_runtime_estimate(
            4,
            2,
            quant_tokens=1024,
            head_dim=64,
            prefix_bf16=64,
            recent_bf16=256,
            max_running_requests=2,
            group_size=64,
        )
        self.assertGreater(two_req, one_req)

    def test_oscar_runtime_default_hp_prefix_pool_has_1024_floor(self) -> None:
        default_pool = kv_bytes_oscar_mixed_runtime_estimate(
            4,
            2,
            quant_tokens=1024,
            head_dim=64,
            prefix_bf16=64,
            recent_bf16=256,
            max_running_requests=1,
            hp_prefix_pool_tokens=None,
            group_size=64,
        )
        explicit_pool = kv_bytes_oscar_mixed_runtime_estimate(
            4,
            2,
            quant_tokens=1024,
            head_dim=64,
            prefix_bf16=64,
            recent_bf16=256,
            max_running_requests=1,
            hp_prefix_pool_tokens=1024,
            group_size=64,
        )
        tiny_pool = kv_bytes_oscar_mixed_runtime_estimate(
            4,
            2,
            quant_tokens=1024,
            head_dim=64,
            prefix_bf16=64,
            recent_bf16=256,
            max_running_requests=1,
            hp_prefix_pool_tokens=64,
            group_size=64,
        )
        self.assertEqual(default_pool, explicit_pool)
        self.assertGreater(default_pool, tiny_pool)


if __name__ == "__main__":
    unittest.main()
