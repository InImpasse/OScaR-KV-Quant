import unittest

from oscar_kv_quant.kv_estimate import (
    kv_bytes_bf16,
    kv_bytes_int2_packed_naive,
    kv_bytes_oscar_mixed_estimate,
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


if __name__ == "__main__":
    unittest.main()
