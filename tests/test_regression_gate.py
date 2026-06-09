import unittest

from oscar_kv_quant.regression_gate import GATE_SCENARIOS, evaluate_gates, resolve_scenario


class RegressionGateTest(unittest.TestCase):
    def _sample_rows(self) -> list[dict[str, str]]:
        return [
            {
                "mode": "bf16",
                "server_ok": "True",
                "peak_mib_delta": "6000",
                "decode_steady_median_tok_s": "65",
                "kv_theory_bf16_gib": "2.5",
                "kv_theory_selected_gib": "2.5",
            },
            {
                "mode": "int2",
                "server_ok": "True",
                "peak_mib_delta": "5800",
                "decode_steady_median_tok_s": "61",
            },
            {
                "mode": "oscar-int2",
                "server_ok": "True",
                "peak_mib_delta": "2000",
                "decode_steady_median_tok_s": "45",
                "decode_flush_median_tok_s": "20",
                "kv_theory_bf16_gib": "2.5",
                "kv_theory_selected_gib": "0.4",
            },
        ]

    def test_balanced_scenario_passes_representative_run(self) -> None:
        scenario = resolve_scenario("balanced", min_memory_ratio=None, min_steady_vs_int2=None, min_steady_vs_bf16=None, min_flush_vs_steady=None)
        results = evaluate_gates(
            self._sample_rows(),
            min_memory_ratio=scenario.min_memory_ratio,
            min_steady_vs_int2=scenario.min_steady_vs_int2,
            min_steady_vs_bf16=scenario.min_steady_vs_bf16,
            min_flush_vs_steady=scenario.min_flush_vs_steady,
        )
        self.assertTrue(all(r.passed for r in results))

    def test_speed_scenario_fails_representative_run(self) -> None:
        scenario = GATE_SCENARIOS["speed"]
        results = evaluate_gates(
            self._sample_rows(),
            min_memory_ratio=scenario.min_memory_ratio,
            min_steady_vs_int2=scenario.min_steady_vs_int2,
            min_steady_vs_bf16=scenario.min_steady_vs_bf16,
            min_flush_vs_steady=scenario.min_flush_vs_steady,
        )
        self.assertFalse(next(r for r in results if r.name == "steady-vs-int2").passed)

    def test_fails_when_oscar_missing(self) -> None:
        rows = [{"mode": "bf16", "server_ok": "True"}]
        scenario = GATE_SCENARIOS["balanced"]
        results = evaluate_gates(
            rows,
            min_memory_ratio=scenario.min_memory_ratio,
            min_steady_vs_int2=scenario.min_steady_vs_int2,
            min_steady_vs_bf16=scenario.min_steady_vs_bf16,
            min_flush_vs_steady=scenario.min_flush_vs_steady,
        )
        self.assertFalse(results[0].passed)

    def test_memory_scenario_skips_speed_when_metrics_missing(self) -> None:
        rows = [
            {
                "mode": "bf16",
                "server_ok": "True",
                "kv_k_size_gb": "1.29",
                "kv_v_size_gb": "1.29",
            },
            {
                "mode": "oscar-int2",
                "server_ok": "True",
                "kv_k_size_gb": "0.19",
                "kv_v_size_gb": "0.19",
                "kv_theory_bf16_gib": "2.5",
                "kv_theory_selected_gib": "0.386",
            },
        ]
        scenario = GATE_SCENARIOS["memory"]
        results = evaluate_gates(
            rows,
            min_memory_ratio=scenario.min_memory_ratio,
            min_steady_vs_int2=scenario.min_steady_vs_int2,
            min_steady_vs_bf16=scenario.min_steady_vs_bf16,
            min_flush_vs_steady=scenario.min_flush_vs_steady,
        )
        self.assertTrue(all(r.passed for r in results), results)


if __name__ == "__main__":
    unittest.main()
