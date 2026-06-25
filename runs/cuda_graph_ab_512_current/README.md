# CUDA Graph A/B 512 Current

This directory keeps the lightweight, machine-readable summary for the
low-risk CUDA graph A/B check. Raw benchmark output remains in the ignored
timestamped run directory:

- `runs/cuda_graph_ab_20260612T062854Z/`

Command used:

```bash
RUN_REAL=1 PROMPT_TOKENS=512 CASES=plain_int2 CASE_TIMEOUT_SEC=60 \
  VRAM_POLL_INTERVAL=0.5 scripts/cuda_graph_ab.sh
```

Result:

- graph off: 2039.0 pp tok/s, 57.2 tg tok/s
- graph on + `GGML_CUDA_GRAPH_OPT=1`: 2020.6 pp tok/s, 55.9 tg tok/s
- delta: -0.90% pp, -2.23% tg

Conclusion: CUDA graph did not improve the 512-token q2 prefill smoke. This is
not evidence for re-running 32k q2.

Validate:

```bash
sha256sum -c runs/cuda_graph_ab_512_current/SHA256SUMS
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```
