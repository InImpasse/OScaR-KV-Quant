# Q2 CUDA Path Current

This directory archives the current no-GPU static report for the llama.cpp CUDA
q2/q4 flash-attention path. It is generated from
`third_party/OSCAR/ggml/src/ggml-cuda/fattn-common.cuh` by:

```bash
python3 scripts/report_q2_cuda_path.py --out-dir runs/q2_cuda_path_current \
  > runs/q2_cuda_path_current/report_stdout.txt
find runs/q2_cuda_path_current -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > runs/q2_cuda_path_current/SHA256SUMS
```

Purpose:

- record q2/q4 KQ/V static path facts without running GPU code;
- capture function-body fingerprints for q2 KQ, q4 KQ, q2 V, q4 V, and dispatch;
- avoid repeating heavy 32k INT2 runs when the relevant q2 CUDA code has not changed.

Current key facts:

- q2 KQ chunk: 3 dp4a, sign/high LUT, exact `m*usum` mean term.
- q4 KQ: 1 dp4a.
- q2 V: scalar q2 decode per V lane.
