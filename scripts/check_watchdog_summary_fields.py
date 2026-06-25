#!/usr/bin/env python3
from combine_llamacpp_kv_runs import FIELDS
from summarize_32k_llamacpp_kv import failure_reason


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required_fields = {"max_peak_mib", "limit_triggered", "exit_code"}
    missing = required_fields - set(FIELDS)
    require(not missing, f"combined CSV missing watchdog fields: {sorted(missing)}")

    reason = failure_reason(
        {
            "limit_triggered": "1",
            "max_peak_mib": "7000",
            "peak_mib": "7024",
            "exit_code": "137",
        },
        has_rates=False,
    )
    require("MAX_PEAK_MIB exceeded" in reason, "watchdog failures should have explicit reason")
    require("limit=7000" in reason and "peak=7024" in reason, "watchdog reason should include peak and limit")

    invalid_json_reason = failure_reason({"exit_code": "124"}, has_rates=False)
    require(
        invalid_json_reason == "missing or invalid llama-bench JSON",
        "missing JSON reason should remain stable for existing q2 NO-GO archives",
    )

    exit_reason = failure_reason({"exit_code": "2"}, has_rates=True)
    require(exit_reason == "exit_code=2", "nonzero exit with rates should keep exit_code reason")

    print("watchdog summary field checks passed")


if __name__ == "__main__":
    main()
