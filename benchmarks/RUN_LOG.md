# Granite 4.0 1B — Run Log

Hardware: WSL2 + RTX 5050 Laptop 8GB (`sm_120`).  
Model: `checkpoints/granite-4.0-1b-base` (symlink to `checkpoints/hf/granite-4.0-1b-base`).  
Rotation: `rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations/`.

Machine-readable summaries: `granite_bench_baseline.json`, `granite_accuracy_baseline.json`.  
Raw CSV/logs: `results/` (gitignored).

---

## Speed / memory — complete (2026-06-08)

Command:

```bash
./scripts/bench_matrix.sh --tag 20260608_fill \
  --presets short,medium,long,32k --modes bf16,int2,oscar-int2
```

Output: `results/granite_bench_matrix/20260608_fill/`

| Prefill | BF16 steady tok/s | plain INT2 | OSCAR INT2 | KV theory (BF16 / OSCAR) |
|---:|---:|---:|---:|---|
| 512 | 46.5 | 33.3 | 50.3 | 0.044 / 0.099 GiB |
| 2048 | 43.1 | 32.8 | 47.3 | 0.161 / 0.116 GiB |
| 8192 | 40.3 | 31.6 | 48.4 | 0.630 / 0.182 GiB |
| 32768 | 29.3 | 24.7 | 36.3 (flush 17.1) | 2.505 / 0.494 GiB |

32K uses `--max-total-tokens 38272` for fair KV pool sizing.  
Regression gate on 32K CSV: balanced scenario passed.

---

## Accuracy — GPQA / GSM8K — complete (2026-06-08)

| Dataset | N | BF16 | plain INT2 | OSCAR INT2 |
|---|---:|---:|---:|---:|
| GPQA | 198 | 23.2% | 15.7% | 24.2% |
| GSM8K | 200 | 56.0% | 3.0% | 54.5% |

OSCAR INT2 tracks BF16; plain INT2 collapses on both tasks.

Run dirs: see `granite_accuracy_baseline.json` → `local_granite.GPQA` / `GSM8K`.

---

## Accuracy — HumanEval suite (2026-06-08 → 2026-06-09)

Command:

```bash
bash rotation/granite-4.0-1b/eval_accuracy_suite.sh \
  --gpqa-num-examples 0 --gsm8k-num-questions 0 \
  --humaneval-num-examples 164 --math-num-examples 500 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations \
  --tag 20260608_fill
```

Log: `results/accuracy_suite_20260608_fill.log`

| Mode | N run | pass@1 | Wall time | Status |
|---|---:|---:|---:|---|
| BF16 | 164 | **31.8%** | ~2.8 h | complete |
| plain INT2 | 164 | **0.0%** | ~5.1 h | complete |
| OSCAR INT2 | 61 / 164 | — | ~7 h then stuck | **failed** |

### OSCAR INT2 failure

At example 61 the SGLang server crashed:

```
ValueError: token_to_kv_pool_allocator memory leak detected!
MIXED_KV_IDLE_LEAK allocator=(...)
```

After crash the eval client retried with `Connection error` (~2100 times in runner log).  
**Partial output is not valid accuracy** — rerun after fixing mixed-KV idle leak or restarting server.

Artifacts:

- BF16: `rotation/granite-4.0-1b/HUMANEVAL/eval_bf16_n164_20260608_fill/metrics.json`
- INT2: `rotation/granite-4.0-1b/HUMANEVAL/eval_int2_n164_20260608_fill/metrics.json`
- OSCAR (invalid): `rotation/granite-4.0-1b/HUMANEVAL/eval_oscar-int2_n164_20260608_fill/`

---

## Accuracy — MATH-500 — not started (full run)

Smoke subset (N=20, 2026-06-08): BF16 5.0%, plain INT2 0.0%, OSCAR INT2 0.0%.  
Full 500-example suite was queued after HumanEval; blocked by OSCAR INT2 failure above.

---

## Not run

- LiveCodeBench v6 (`eval_lcb_v6_granite.sh`)
- AIME 25 (`eval_simple_suite_granite.sh --task aime25`)

---

## Refresh commands

```bash
# Bench matrix
./scripts/bench_matrix.sh --tag $(date +%Y%m%d)

# Accuracy (full Granite suite)
bash rotation/granite-4.0-1b/eval_accuracy_suite.sh \
  --gpqa-num-examples 198 --gsm8k-num-questions 200 \
  --humaneval-num-examples 164 --math-num-examples 500 \
  --rot-dir rotation/granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations
```

After new runs, update `granite_*_baseline.json` and this log.
