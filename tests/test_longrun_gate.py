import unittest

from oscar_kv_quant.longrun_gate import evaluate_longrun_gate, evaluate_longrun_many_gate


class LongrunGateTest(unittest.TestCase):
    def _baseline(self) -> dict:
        return {
            "ok": True,
            "failures": 0,
            "completed_requests": 4,
            "num_requests": 4,
            "log_errors": [],
            "peak_mib_total": 6209,
            "profile": "granite",
            "prefill_tokens": 32768,
            "max_new_tokens": 128,
            "kv_k_size_gb": 1.29,
            "kv_v_size_gb": 1.29,
            "request_tok_s_median": 8.78,
            "request_tok_s_first_half_median": 7.17,
            "request_tok_s_second_half_median": 8.785,
            "server_decode_steady_median_tok_s": 36.76,
            "cached_prefill_cache_ratio_median": 0.996,
            "requests": [
                {"ok": True, "tok_s": 4.85},
                {"ok": True, "tok_s": 9.49},
                {"ok": True, "tok_s": 8.82},
                {"ok": True, "tok_s": 8.75},
            ],
        }

    def _oscar(self) -> dict:
        return {
            "ok": True,
            "failures": 0,
            "completed_requests": 4,
            "num_requests": 4,
            "log_errors": [],
            "peak_mib_total": 5883,
            "profile": "granite",
            "prefill_tokens": 32768,
            "max_new_tokens": 128,
            "selected_vs_bf16_kv_ratio": 0.184,
            "request_tok_s_median": 11.67,
            "request_tok_s_first_half_median": 8.43,
            "request_tok_s_second_half_median": 14.65,
            "server_decode_steady_median_tok_s": 51.26,
            "cached_prefill_cache_ratio_median": 0.991,
            "requests": [
                {"ok": True, "tok_s": 5.63},
                {"ok": True, "tok_s": 11.23},
                {"ok": True, "tok_s": 12.11},
                {"ok": True, "tok_s": 17.19},
            ],
        }

    def test_passes_representative_32k_128_run(self) -> None:
        results = evaluate_longrun_gate(self._oscar(), self._baseline())
        self.assertTrue(all(r.passed for r in results), results)

    def test_fails_when_oscar_peak_is_not_lower(self) -> None:
        oscar = self._oscar()
        oscar["peak_mib_total"] = 6300
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(next(r for r in results if r.name == "peak-memory-lower").passed)

    def test_fails_when_tail_speed_regresses(self) -> None:
        oscar = self._oscar()
        oscar["requests"][-2:] = [{"ok": True, "tok_s": 5.0}, {"ok": True, "tok_s": 5.5}]
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(next(r for r in results if r.name == "request-tail-speed").passed)

    def test_fails_on_log_errors(self) -> None:
        oscar = self._oscar()
        oscar["log_errors"] = ["RuntimeError: boom"]
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(next(r for r in results if r.name == "oscar-log-clean").passed)

    def test_fails_on_workload_mismatch(self) -> None:
        baseline = self._baseline()
        baseline["max_new_tokens"] = 64
        results = evaluate_longrun_gate(self._oscar(), baseline)
        self.assertFalse(next(r for r in results if r.name == "same-max-new-tokens").passed)

    def test_fails_when_second_half_slows_down(self) -> None:
        oscar = self._oscar()
        oscar["request_tok_s_first_half_median"] = 10.0
        oscar["request_tok_s_second_half_median"] = 8.0
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(
            next(r for r in results if r.name == "oscar-second-half-stability").passed
        )

    def test_reports_last_quarter_stability_separately(self) -> None:
        oscar = self._oscar()
        oscar["request_tok_s_first_half_median"] = 10.0
        oscar["request_tok_s_second_half_median"] = 8.0
        oscar["requests"] = [
            {"ok": True, "tok_s": 10.0},
            {"ok": True, "tok_s": 10.0},
            {"ok": True, "tok_s": 6.0},
            {"ok": True, "tok_s": 10.0},
        ]
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(
            next(r for r in results if r.name == "oscar-second-half-stability").passed
        )
        self.assertTrue(
            next(r for r in results if r.name == "oscar-last-quarter-stability").passed
        )

    def test_fails_when_last_quarter_slows_down(self) -> None:
        oscar = self._oscar()
        oscar["request_tok_s_first_half_median"] = 10.0
        oscar["request_tok_s_second_half_median"] = 9.5
        oscar["requests"][-1] = {"ok": True, "tok_s": 8.0}
        results = evaluate_longrun_gate(oscar, self._baseline())
        self.assertFalse(
            next(r for r in results if r.name == "oscar-last-quarter-stability").passed
        )

    def test_many_gate_requires_worst_run_to_clear_baseline(self) -> None:
        fast = self._oscar()
        slow = self._oscar()
        slow["request_tok_s_median"] = 8.0
        baseline = self._baseline()
        baseline["request_tok_s_median"] = 9.0
        results = evaluate_longrun_many_gate([fast, slow], [baseline])
        self.assertFalse(
            next(r for r in results if r.name == "request-median-speed-all").passed
        )

    def test_many_gate_passes_representative_repeats(self) -> None:
        oscar_a = self._oscar()
        oscar_b = self._oscar()
        oscar_b["request_tok_s_median"] = 12.0
        baseline = self._baseline()
        results = evaluate_longrun_many_gate([oscar_a, oscar_b], [baseline])
        self.assertTrue(all(r.passed for r in results), results)


if __name__ == "__main__":
    unittest.main()
