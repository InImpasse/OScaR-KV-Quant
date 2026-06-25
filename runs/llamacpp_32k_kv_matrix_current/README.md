# llama.cpp 32k KV Matrix

This directory archives the current llama.cpp-only KV benchmark evidence for the
Granite 4.0 1B BF16 model. It is intentionally unignored in `.gitignore` while
other `runs/` outputs remain ignored.

## Contents

- `combined.csv` / `combined.md`: merged matrix used by docs and checks.
- `raw/`: archived per-case run directories copied from `/tmp/llamacpp_*`.
- `SHA256SUMS`: checksums for the archived evidence files.

The raw directories include each case's `summary.csv`, `summary.md`, per-case
JSON/stdout/stderr, VRAM metrics, and config metadata. The total archive is small
enough to keep in-repo.

## Rebuild Combined Report

```bash
python3 scripts/combine_llamacpp_kv_runs.py \
  --out-dir runs/llamacpp_32k_kv_matrix_current \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_32k_bf16_current \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_32k_oscar_int4_current \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_32k_plain_int4_current \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_16k_plain_int2_240s \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_16k_oscar_int2_240s \
  runs/llamacpp_32k_kv_matrix_current/raw/llamacpp_32k_oscar_int2_480s
```

Validate conclusions without running GPU work:

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

Verify archived files:

```bash
sha256sum -c runs/llamacpp_32k_kv_matrix_current/SHA256SUMS
```

## Current Conclusion

- 32k BF16 and 32k INT4 are valid.
- INT4 peak VRAM savings closely match theoretical KV savings.
- 16k plain/oscar INT2 are valid but slow.
- 32k OSCAR INT2 timed out with empty JSON and remains NO-GO for the current
  exact q2 CUDA path.
