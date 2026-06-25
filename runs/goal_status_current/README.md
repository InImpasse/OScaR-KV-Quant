# Goal Status Current

This directory archives the current machine-readable audit of progress toward
the 32k llama.cpp KV goal.

Regenerate:

```bash
python3 scripts/audit_goal_status.py --out-dir runs/goal_status_current
find runs/goal_status_current -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > runs/goal_status_current/SHA256SUMS
```

Validate:

```bash
sha256sum -c runs/goal_status_current/SHA256SUMS
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

Current summary:

- BF16 32k baseline: complete.
- INT4 32k memory and speed: complete.
- 16k INT2 gate: complete.
- CUDA graph 512 A/B: complete, no speedup.
- 32k INT2 speed target: incomplete.
