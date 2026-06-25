#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fattn_common = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-common.cuh").read_text()
    fused = (ROOT / "third_party/OSCAR/ggml/src/ggml-cuda/fattn-q2_0-f16.cu").read_text()
    status = (ROOT / "docs/KV_CACHE_CURRENT_STATUS.md").read_text()
    triage = (ROOT / "docs/Q2_ROTATION_ACCURACY_TRIAGE_20260614.md").read_text()

    q2_kq = fattn_common.split("vec_dot_fattn_vec_KQ_q2_0_chunk", 1)[1].split("template<int D, int nthreads>", 1)[0]
    q2_v = fattn_common.split("dequantize_V_q2_0", 1)[1].split("template <typename T, int ne>", 1)[0]

    require("q2_0_dequantize_scalar_cuda" in q2_v,
            "generic CUDA q2 V path should currently be direct scalar decode")
    require("q2_0_dequantize_row_owht_cuda" not in q2_kq and "q2_0_dequantize_row_owht_cuda" not in q2_v,
            "generic CUDA q2/q2 vector FA must not be mistaken for OWHT-aware reader")
    require("q2_0_dequantize_row_owht_cuda<D>" in fused and "use_owht" in fused,
            "fused q2_0+F16 HP kernel should be the only staged OWHT-aware CUDA attention reader")
    require("generic vector FA kernels" in status and "not yet apply inverse OWHT" in status,
            "status doc must explain generic q2 vector FA lacks inverse OWHT for Hadamard-applied writes")
    require("LLAMA_KV_NO_HADAMARD=1" in status and "direct scalar reader is compatible" in status,
            "status doc must distinguish compatible no-Hadamard q2 from Hadamard-applied OWHT")
    require("deeper" in triage and "K/V clip ratio" in triage,
            "triage doc must record split-clip probe did not rescue q2")
    require("matching inverse-OWHT reader" in triage and "staged inverse-OWHT reader" in triage,
            "triage doc must record q2 OWHT reader limitation")

    print("q2 OWHT reader limit checks passed")


if __name__ == "__main__":
    main()
