import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from oscar_kv_quant.log_metrics import parse_server_log


class LogMetricsTest(unittest.TestCase):
    def test_parse_decode_and_prefill_breakdown(self) -> None:
        sample = """
[2026-06-05 16:22:46] Enable unified mixed KV (int2): prefix=64 recent=256
[2026-06-05 16:22:50] KV Cache is allocated. #tokens: 32832, K size: 0.39 GB, V size: 0.39 GB
[2026-06-05 16:22:55] Prefill batch, input throughput (token/s): 100.0
[2026-06-05 16:22:56] Prefill batch, input throughput (token/s): 4000.0
[2026-06-05 16:23:07] Decode batch, gen throughput (token/s): 1.62
[2026-06-05 16:23:20] Decode batch, gen throughput (token/s): 26.60
[2026-06-05 16:23:34] Decode batch, gen throughput (token/s): 3.40
[2026-06-05 16:23:35] Decode batch, gen throughput (token/s): 24.77
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(sample, encoding="utf-8")
            metrics = parse_server_log(
                path,
                measurement_requests=1,
                decode_tokens_per_request=64,
            )

        self.assertEqual(metrics.kv_pool_tokens, 32832)
        self.assertAlmostEqual(metrics.kv_k_size_gb or 0.0, 0.39)
        self.assertTrue(metrics.unified_mixed_kv)
        self.assertEqual(metrics.decode_max_tok_s, 26.60)
        self.assertEqual(metrics.decode_first_tok_s, 1.62)
        self.assertIsNotNone(metrics.decode_steady_median_tok_s)
        self.assertGreater(metrics.decode_steady_median_tok_s or 0.0, 20.0)
        self.assertIsNotNone(metrics.decode_flush_median_tok_s)
        self.assertLess(metrics.decode_flush_median_tok_s or 999.0, 10.0)
        self.assertAlmostEqual(metrics.prefill_wall_s or 0.0, 1.0, places=0)
        self.assertAlmostEqual(metrics.decode_wall_s or 0.0, 28.0, places=0)
        self.assertIsNotNone(metrics.flush_step_fraction)
        self.assertGreater(metrics.flush_step_fraction or 0.0, 0.0)
        self.assertIsNotNone(metrics.effective_decode_tok_s)
        self.assertGreater(metrics.effective_decode_tok_s or 0.0, 1.0)

    def test_wall_times_use_last_measurement_window(self) -> None:
        base = datetime(2026, 6, 5, 16, 0, 0)
        lines = []
        # Warmup request: short prefill/decode
        lines.append(
            f"[{(base).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Prefill batch, input throughput (token/s): 1000.0"
        )
        lines.append(
            f"[{(base + timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Decode batch, gen throughput (token/s): 40.0"
        )
        # Measured request
        lines.append(
            f"[{(base + timedelta(seconds=10)).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Prefill batch, input throughput (token/s): 2000.0"
        )
        lines.append(
            f"[{(base + timedelta(seconds=12)).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Prefill batch, input throughput (token/s): 3000.0"
        )
        lines.append(
            f"[{(base + timedelta(seconds=20)).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Decode batch, gen throughput (token/s): 50.0"
        )
        lines.append(
            f"[{(base + timedelta(seconds=22)).strftime('%Y-%m-%d %H:%M:%S')}] "
            "Decode batch, gen throughput (token/s): 48.0"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            metrics = parse_server_log(
                path,
                measurement_requests=1,
                decode_tokens_per_request=32,
            )
        self.assertAlmostEqual(metrics.prefill_wall_s or 0.0, 2.0)
        self.assertAlmostEqual(metrics.decode_wall_s or 0.0, 2.0)
        self.assertAlmostEqual(metrics.effective_decode_tok_s or 0.0, 16.0)

    def test_parse_multiple_mixed_kv_fields_on_one_line(self) -> None:
        sample = """
[2026-06-08 01:59:42] Enable unified mixed KV (int2): prefix=64 recent=256 flush_interval=8 num_quant_pages=1145 N_Q=8 hp_dtype=bfloat16 scale_dtype=bfloat16 max_total_num_tokens=9152 max_req_slots=1 hp_prefix_pool_tokens=1024
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(sample, encoding="utf-8")
            metrics = parse_server_log(path)

        self.assertTrue(metrics.unified_mixed_kv)
        self.assertEqual(metrics.max_total_num_tokens, 9152)
        self.assertEqual(metrics.hp_prefix_pool_tokens, 1024)

    def test_parse_cached_prefill_token_summary(self) -> None:
        sample = """
[2026-06-08 03:06:19] Prefill batch, #new-seq: 1, #new-token: 272, #cached-token: 29520, token usage: 0.85, input throughput (token/s): 85.97
[2026-06-08 03:06:26] Prefill batch, #new-seq: 1, #new-token: 272, #cached-token: 29520, token usage: 0.85, input throughput (token/s): 36.71
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(sample, encoding="utf-8")
            metrics = parse_server_log(path)

        self.assertEqual(metrics.prefill_new_tokens, [272, 272])
        self.assertEqual(metrics.prefill_cached_tokens, [29520, 29520])
        self.assertEqual(metrics.cached_prefill_new_median_tokens, 272)
        self.assertEqual(metrics.cached_prefill_cached_median_tokens, 29520)
        self.assertGreater(metrics.cached_prefill_cache_ratio_median or 0.0, 0.99)

    def test_request_windows_capture_prefill_and_decode_per_request(self) -> None:
        sample = """
[2026-06-08 03:06:19] Prefill batch, #new-seq: 1, #new-token: 80, #cached-token: 29712, token usage: 0.85, input throughput (token/s): 5.00
[2026-06-08 03:06:20] Decode batch, #running-req: 1, #token: 29560, token usage: 0.85, cuda graph: True, gen throughput (token/s): 3.00, #queue-req: 0
[2026-06-08 03:06:21] Decode batch, #running-req: 1, #token: 29600, token usage: 0.85, cuda graph: True, gen throughput (token/s): 45.00, #queue-req: 0
[2026-06-08 03:06:26] Prefill batch, #new-seq: 1, #new-token: 80, #cached-token: 29712, token usage: 0.85, input throughput (token/s): 10.00
[2026-06-08 03:06:27] Decode batch, #running-req: 1, #token: 29560, token usage: 0.85, cuda graph: True, gen throughput (token/s): 6.00, #queue-req: 0
[2026-06-08 03:06:28] Decode batch, #running-req: 1, #token: 29600, token usage: 0.85, cuda graph: True, gen throughput (token/s): 50.00, #queue-req: 0
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(sample, encoding="utf-8")
            metrics = parse_server_log(path)

        self.assertEqual(len(metrics.request_windows), 2)
        self.assertEqual(metrics.request_windows[0].prefill_new_tokens, 80)
        self.assertEqual(metrics.request_windows[0].prefill_cached_tokens, 29712)
        self.assertEqual(metrics.request_windows[0].prefill_tok_s, 10.0)
        self.assertEqual(metrics.request_windows[0].prefill_interval_est_s, 8.0)
        self.assertEqual(metrics.request_windows[0].prefill_wall_est_s, 8.0)
        self.assertEqual(metrics.request_windows[0].decode_first_tok_s, 3.0)
        self.assertEqual(metrics.request_windows[0].decode_steady_median_tok_s, 45.0)
        self.assertIsNone(metrics.request_windows[1].prefill_tok_s)
        self.assertIsNone(metrics.request_windows[1].prefill_interval_est_s)
        self.assertIsNone(metrics.request_windows[1].prefill_wall_est_s)

    def test_decode_classification_uses_high_plateau_for_bimodal_short_runs(self) -> None:
        sample = "\n".join(
            [
                f"[2026-06-08 03:00:{i:02d}] Decode batch, gen throughput (token/s): {v}"
                for i, v in enumerate(
                    [0.72, 3.47, 44.49, 4.57, 7.33, 43.47, 7.39, 55.49]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(sample + "\n", encoding="utf-8")
            metrics = parse_server_log(path)

        self.assertGreater(metrics.decode_steady_median_tok_s or 0.0, 40.0)
        self.assertLess(metrics.decode_flush_median_tok_s or 999.0, 10.0)


if __name__ == "__main__":
    unittest.main()
