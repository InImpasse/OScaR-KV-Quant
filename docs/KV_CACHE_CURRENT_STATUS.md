# KV Cache Current Status

## Current Result

The current CUDA path builds and links with `q2_0` KV cache support:

- CUDA `SET_ROWS` can write `q2_0` KV blocks.
- CUDA flash attention vector kernels can consume `q2_0` K/V directly.
- CUDA fused LP+HP attention is named as `q2_0` + F16: `ggml_flash_attn_ext_q2_0_f16`.
- CPU, CUDA, and Metal keep the same public storage type name: `q2_0`.
- CPU, CUDA, and Metal use shared `q2_0` Lloyd-Max centroid/threshold constants
  from `ggml/src/ggml-common.h`, avoiding backend drift in INT2 reconstruction.
- Metal `q2_0` dequantization now uses the same Lloyd-Max centroids as CPU and
  CUDA (`-0.9816`, `-0.4528`, `0.4528`, `0.9816`) instead of reconstructing
  with raw code values `0..3`.

Important semantic detail: CPU `q2_0` does more than centroid decode by default.
Its reference path groups up to 128 dimensions, subtracts a group mean, applies
OWHT, then Lloyd-Max quantizes 32-element sub-blocks. CPU dequant and CPU
`vec_dot` undo the OWHT before using the values. CUDA and Metal now share the
same public `block_q2_0` name and the same centroid table, but their FA helpers
still directly decode `m + d * centroid(code)` and do not yet apply inverse
OWHT. That is the main remaining implementation mismatch.

CUDA now has staged OWHT-aware helper code for `q2_0` group quant/dequant and
the fused `q2_0` + F16 HP attention path can use it under
`LLAMA_KV_Q2_0_OWHT=1` (`LLAMA_CUDA_Q2_0_OWHT=1` remains a compatibility
alias). The staged CUDA helper follows the CPU grouping rule:
128-dim groups for head dimensions at least 128, otherwise 32-dim per-block
groups. It also follows the CPU `LLAMA_KV_NO_HADAMARD=1` gate: the staged path
still uses group mean + Lloyd-Max packing, but skips the Hadamard stage when the
calibrated rotation already includes it. The staged CUDA writer also applies the
same `LLAMA_KV_CLIP_RATIO` percentile clamp as CPU when the ratio is in `(0, 1)`.
The gated fused HP path now decodes each low-precision K/V row once into a local
float buffer before dot/value use, instead of repeating inverse OWHT for each
scalar element. The heavier OWHT helpers live in CUDA `q2_0-owht.cuh` and are
included only by `SET_ROWS` and the fused HP kernel; the default direct `q2_0`
vector FA path continues to include the lightweight centroid helper only.
This is deliberately not enabled by default yet. There are two distinct cases:
with `LLAMA_KV_NO_HADAMARD=1`, the staged q2 writer stores the same direct
`m + d * centroid(code)` values that the generic CUDA scalar reader consumes, so
the direct scalar reader is compatible with the no-Hadamard staged layout. With
Hadamard-applied OWHT writes, however, the generic vector FA kernels still do
not yet apply inverse OWHT; plain vector FA needs a high-performance matching
inverse-OWHT reader for both K and V before those writes can be enabled safely.
Enabling those writes without matching all readers would corrupt plain `q2_0`
attention. The active default remains the previously benchmarked direct centroid
path.

## Memory And Speed Summary

Local RTX 5050 Laptop measurements show that KV compression is most visible when
KV cache is a meaningful part of total VRAM. On Granite 1B, long prompts show a
clear reduction: `f16` peak was about 4032 MiB and `q2_0` about 3766 MiB at the
recorded long setting. On Gemma E2B, the measured peak was dominated by model
weights and fixed runtime allocations, so `q2_0` barely reduced total peak VRAM
in the short/medium/long rows.

The existing matrix run has been summarized here:
`runs/q2hp_bench_matrix_20260604T164856Z/kv_matrix_summary.md`.
The measured-vs-theoretical joined report is here:
`runs/q2hp_bench_matrix_20260604T164856Z/kv_matrix_combined.md`.
Key ratios from that run:

- Granite long: `q2_0` saved 266 MiB vs F16 peak (93.4% of F16 peak), prompt
  processing was 24.9% of F16 speed, and generation was 101.8% of F16 speed.
- Granite medium: `q2_0` saved 126 MiB (96.7% of F16 peak), prompt processing
  was 41.3% of F16, and generation was 105.9% of F16.
- Gemma short/medium/long: total peak was 99.4% / 99.2% / 99.9% of F16, so
  total VRAM improvement was not meaningful in this setup. Generation speed was
  about 58-68% of F16.

Theoretical KV storage has also been estimated from GGUF metadata:
`runs/kv_theoretical_512_2048_4096_hp.csv`.

For both local models, the KV element geometry is different but the storage
ratio by cache type is the same:

- `q8_0`: about 53.1% of F16 KV storage.
- `q4_0`: about 28.1% of F16 KV storage.
- `q2_0`: about 18.8% of F16 KV storage.

At 4096 tokens, Granite F16 KV is about 320 MiB and `q2_0` is about 60 MiB,
matching the measured 266 MiB total-VRAM reduction closely. Gemma F16 KV is
about 280 MiB and `q2_0` about 52.5 MiB, but the measured total peak barely
moved because model weights and allocator/runtime overhead dominate the 5.2 GiB
peak in this test.

The joined report makes this more explicit:

- Granite `q2_0` measured savings track theoretical KV savings closely:
  36 MiB measured vs 32.5 MiB theoretical at 512 tokens, 126 vs 130 MiB at
  2048 tokens, and 266 vs 260 MiB at 4096 tokens.
- Gemma `q2_0` does not show the same total-peak behavior: it measured 30 MiB
  saved vs 28.4 MiB theoretical at 512 tokens, but only 44 MiB vs 113.8 MiB at
  2048 tokens and 4 MiB vs 227.5 MiB at 4096 tokens. This means the recorded
  peak for Gemma is not dominated by KV storage at those settings.
- `q2_0_hp` rows need separate interpretation. The theoretical table includes
  the fixed F16 HP side cache, but the recorded matrix rows have the same peak
  as plain `q2_0`. That is useful for spotting that the current sampled peak did
  not expose the HP side-buffer cost, but it is not enough to claim HP mode is
  memory-equivalent to plain `q2_0`.

With `LLAMA_KV_HP_SINK=512` and `LLAMA_KV_HP_RECENT=2048`, `q2_0_hp` is not a
memory-minimum mode for short/medium contexts: it adds a fixed 2560-token F16 HP
side cache. In the theoretical table, `q2_0_hp` is larger than F16 at 512 and
2048 tokens, and only becomes smaller than F16 at 4096 tokens.

Speed is mixed:

- Prompt processing is slower with `q2_0`, because the current path quantizes
  KV writes and dequantizes inside attention.
- Granite token generation is roughly similar to `f16`, sometimes slightly
  better in the recorded rows.
- Gemma token generation is slower with `q2_0`, suggesting its shape/kernel mix
  is less favorable for the current fused path.
- `q2_0` + F16 HP is useful for preserving sink/recent precision, but it is not
  a pure memory-minimum mode because the HP side cache is F16.

## How To Re-run The Matrix

Before running benchmark or PPL matrices, check the local environment:

```bash
./scripts/check_kv_env.sh
```

The preflight checks model files, required binaries, `nvidia-smi`, CUDA device
initialization, and whether the local `llama-bench` exposes the expected `-p/-n`
benchmark arguments. In the default sandbox, GPU/NVML access can be blocked. With
elevated execution on this machine, `nvidia-smi` and `llama-bench --list-devices`
successfully see the RTX 5050 Laptop GPU.

At the time of the latest smoke run, another `/llama-cli` process was already
using about 5.3 GiB of VRAM. Full VRAM conclusions should therefore be collected
on an otherwise idle GPU, or interpreted through the per-row `baseline_mib` and
`delta_mib` fields rather than total peak alone.

Elevated GPU smoke on the RTX 5050 Laptop GPU now works. A tiny Granite-only
run (`prompt=128`, `gen=32`, `f16,q2_0`) completed successfully with stable
baseline 4425 MiB:

| model | prompt/gen | KV | delta MiB | pp tok/s | tg tok/s |
|---|---:|---|---:|---:|---:|
| granite | 128/32 | f16 | 3295 | 1648.5 | 68.1 |
| granite | 128/32 | q2_0 | 3281 | 830.9 | 58.4 |

This is only a smoke test, not the final matrix: at such a short context q2_0
saves only 14 MiB of incremental VRAM because the model/runtime footprint
dominates. Use the existing 512/2048/4096 matrix or rerun the full matrix on an
idle GPU for real memory conclusions.

`bench_kv_cache_matrix.sh` and `run_kv_ppl_matrix.sh` now default to
`DRY_RUN=1`, so they print commands before touching preflight, corpus checks, or
the GPU. Real benchmark matrix runs require `DRY_RUN=0 ACK_MATRIX_BENCH=1`; real
PPL matrix runs require `DRY_RUN=0 ACK_PPL_MATRIX=1`. When running for real, both
scripts guard formal runs against a busy GPU. By default these scripts refuse to
start if the current `nvidia-smi` baseline exceeds `MAX_BASELINE_MIB=1024`. For
quick smoke tests on a known-busy GPU, set `ALLOW_BUSY_GPU=1`; those runs should
not be used as final VRAM evidence. For CPU-only or quality-only PPL checks, set
`MEASURE_VRAM=0`.

When the guard refuses a run, it now prints the current GPU memory baseline and
the raw `nvidia-smi --query-compute-apps=pid,process_name,used_memory` snapshot.
Benchmark runs also write `guard_baseline_mib` and the process snapshot into
`config.txt`, even when `ALLOW_BUSY_GPU=1` is used for a smoke run. On the
current machine the busy snapshot has been:

```text
guard_baseline_mib=4425
312710, [Not Found], [N/A]
```

This confirms that busy-GPU smoke rows should be interpreted through execution
success only, not as final total-VRAM evidence.

To run a formal matrix as soon as the GPU is idle, use:

```bash
WAIT_FOR_IDLE_GPU=1 \
GPU_IDLE_TIMEOUT_SEC=3600 \
GPU_IDLE_POLL_SEC=10 \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

Use the same `WAIT_FOR_IDLE_GPU` variables with `run_kv_ppl_matrix.sh` when
collecting PPL with VRAM sampling.

The benchmark and PPL matrix scripts run this preflight by default. To
intentionally bypass it, set `RUN_PREFLIGHT=0`. The PPL matrix can also skip
VRAM sampling with `MEASURE_VRAM=0`; the speed matrix always samples VRAM via
`measure_vram.sh` and therefore requires working `nvidia-smi`.

The preflight can be scoped for targeted checks:

```bash
CHECK_BENCH=0 CHECK_PPL=1 CHECK_GPU=0 ./scripts/check_kv_env.sh
CHECK_BENCH=1 CHECK_PPL=0 CHECK_GPU=1 ./scripts/check_kv_env.sh
```

`run_kv_ppl_matrix.sh` uses the PPL-only scope, and skips GPU checks
automatically when `MEASURE_VRAM=0`. `bench_kv_cache_matrix.sh` uses the
benchmark+GPU scope because it always measures VRAM.

PPL rows record real command status, duration, stdout, and stderr. The PPL
matrix also treats a row as failed when `llama-perplexity` exits successfully
but does not emit a parseable `PPL = ...` line; this catches invalid corpora
that are too short for the selected context. The same check is enabled whether
or not VRAM sampling is active.

Use the matrix script for two models, short/medium/long prompts, multiple KV
types, speed, and sampled VRAM:

```bash
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

Useful overrides:

```bash
LLAMA_BENCH=third_party/OSCAR/build-cuda/bin/llama-bench \
LENGTHS=short:512,medium:2048,long:4096 \
KV_MODES=f16,q8_0,q4_0,q2_0,q2_0_hp \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

By default `KV_MODES` applies the same type to K and V. To test asymmetric K/V
cache choices, use `KV_PAIRS` with `K/V` entries:

```bash
KV_PAIRS=f16/f16,q8_0/q2_0,q2_0/q8_0,q2_0/q2_0 \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

The label for an asymmetric row is normalized as `k<K>_v<V>`, for example
`kq8_0_vq2_0`. The summarizer reports `KV`, `K cache`, and `V cache` columns so
these rows remain comparable with same-type rows.

For CUDA q2_0 implementation ablations, the matrix scripts also accept explicit
staged-mode labels:

| mode label | cache type passed to llama.cpp | extra env |
|---|---|---|
| `q2_0` | `q2_0` | direct centroid path |
| `q2_0_hp` | `q2_0` | `LLAMA_KV_HP_SINK/RECENT` |
| `q2_0_owht` | `q2_0` | `LLAMA_KV_Q2_0_OWHT=1`, Hadamard on, clip off |
| `q2_0_owht_nohad` | `q2_0` | staged group path with `LLAMA_KV_NO_HADAMARD=1`, clip off |
| `q2_0_owht_clip` | `q2_0` | staged group path with Hadamard on and `LLAMA_KV_CLIP_RATIO` |
| `q2_0_owht_nohad_clip` | `q2_0` | staged group path with `LLAMA_KV_NO_HADAMARD=1` and `LLAMA_KV_CLIP_RATIO` |

These are measurement labels, not new public GGML storage types. CPU, CUDA, and
Metal still use the shared public cache type name `q2_0`; the labels only make
the CUDA staged writer/reader environment explicit in benchmark and PPL output.
The clip ratio defaults to `Q2_0_CLIP_RATIO=0.96` and can be overridden when
running the scripts. The benchmark summarizer treats staged `q2_0_owht*` labels
as storage-equivalent to `q2_0` when joining theoretical KV storage, so existing
`q2_0` theory CSVs can still be reused.

`llama-bench` sizes each row from `-p/--n-prompt` and `-n/--n-gen`; the matrix
script records `CONTEXT` in `config.txt` for compatibility with older runs, but
does not pass `-c` because this `llama-bench` build has no context-size option.

Outputs are written to `runs/kv_matrix_<UTC>/`. Each row has:

- `<label>.json`: `llama-bench` JSON output.
- `<label>.metrics.tsv`: sampled VRAM over time.
- `<label>.summary.txt`: exit code, duration, baseline/peak VRAM, K/V cache
  types, and the per-row `LLAMA_KV_*` environment values used by staged q2_0
  modes.
- `config.txt`: exact run configuration.

Summarize any matrix directory into CSV and Markdown:

```bash
./scripts/summarize_kv_matrix.py runs/kv_matrix_<UTC>
```

The summary reports peak VRAM, MiB saved vs same-model/same-length F16, prompt
processing tok/s, generation tok/s, and relative speed ratios. It also reports
the sampled baseline, per-command delta VRAM, actual K/V cache type, and staged
q2_0 env metadata. Use delta columns when other GPU processes or allocator
residue make the absolute peak baseline unstable.

To join measured peak/speed with theoretical KV storage, pass the estimate CSV:

```bash
./scripts/summarize_kv_matrix.py \
  runs/kv_matrix_<UTC> \
  --theory-csv runs/kv_theoretical_512_2048_4096_hp.csv \
  --out-prefix runs/kv_matrix_<UTC>/kv_matrix_combined
```

The joined summary adds theoretical KV MiB, theoretical KV savings vs F16, and
the measured-total-peak savings as a percentage of theoretical KV savings.

Gate benchmark rows against memory and speed targets:

```bash
./scripts/check_kv_bench_gate.py \
  runs/kv_matrix_<UTC>/kv_matrix_combined.csv \
  --model granite \
  --length long \
  --kv q2_0 \
  --min-delta-saved-mib 200 \
  --min-tg-ratio 0.95 \
  --min-measured-over-theory 0.8 \
  --fail-empty
```

On the existing matrix, Granite long `q2_0` passes this gate: it saved 266 MiB
of incremental VRAM, generation speed was 101.8% of F16, and measured savings
tracked theoretical KV savings. Gemma long `q2_0` fails the same gate: it saved
only 4 MiB, generation speed was 58.3% of F16, and measured savings were only
1.76% of theoretical KV savings. This captures the current asymmetry: Granite
shows useful long-context KV-cache compression; Gemma still needs profiling and
kernel/allocator investigation before calling the optimization effective there.

Estimate theoretical KV-cache storage from GGUF metadata:

```bash
./scripts/estimate_kv_cache.py \
  --model granite:checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
  --model gemma:checkpoints/gguf/gemma-4-E2B-it-bf16.gguf \
  --contexts 512,2048,4096 \
  --kv-types f16,q8_0,q4_0,q2_0,q2_0_hp,q8_0/q2_0,q2_0/q8_0 \
  --hp-sink 512 \
  --hp-recent 2048 \
  --out runs/kv_theoretical_512_2048_4096_hp.csv
```

## How To Run Quality Validation

Throughput and VRAM are not enough for this optimization. The added PPL matrix
script evaluates the same models and KV modes with `llama-perplexity`, optionally
sampling VRAM through the same `measure_vram.sh` wrapper.

Build the tool if needed:

```bash
cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-perplexity
```

For a deterministic local smoke corpus, generate a repository-derived text file:

```bash
./scripts/make_ppl_smoke_corpus.sh /tmp/oscar_kv_ppl_smoke_corpus.txt
```

This corpus is only for execution and regression sanity checks; it is not a
standard language-model quality benchmark. The PPL matrix records
`corpus_bytes` and `corpus_sha256` in `config.txt` so repeated runs can be tied
to the exact same input file.

Run a short quality matrix with a local text corpus:

```bash
CORPUS=/path/to/wiki.test.raw \
CHUNKS=8 \
CONTEXTS=short:512,medium:2048,long:4096 \
KV_MODES=f16,q8_0,q4_0,q2_0,q2_0_hp \
DRY_RUN=0 ACK_PPL_MATRIX=1 \
./scripts/run_kv_ppl_matrix.sh
```

As with the throughput matrix, `KV_PAIRS` can be used for asymmetric K/V PPL
checks:

```bash
CORPUS=/path/to/wiki.test.raw \
KV_PAIRS=f16/f16,q8_0/q2_0,q2_0/q8_0,q2_0/q2_0 \
DRY_RUN=0 ACK_PPL_MATRIX=1 \
./scripts/run_kv_ppl_matrix.sh
```

Summarize the result:

```bash
./scripts/summarize_kv_ppl.py runs/kv_ppl_<UTC>
```

Gate the summarized rows against same-model/same-context F16 baselines:

```bash
./scripts/check_kv_ppl_gate.py \
  runs/kv_ppl_<UTC>/kv_ppl_summary.csv \
  --max-ratio 1.05
```

The gate fails rows with non-zero exits, missing PPL, missing ratios, or PPL
ratio above the configured threshold. This is intended as the default quality
check before enabling CUDA OWHT or other q2_0 reader/writer changes by default.
Use `--max-delta` when a corpus has a known acceptable absolute PPL window, and
`--allow-kv` to gate only specific KV labels during ablations.

The summary reports command exit code, PPL, PPL delta/ratio vs
same-model/same-context F16, sampled peak VRAM, and peak savings. This is the
main regression gate for future CUDA OWHT changes: a direct CUDA q2_0 speed
improvement is not acceptable unless PPL stays within the expected loss envelope
vs F16 and does not regress relative to the previous q2_0 path.

PPL summaries also include per-row K/V cache type and staged q2_0 env metadata,
so a row labeled `q2_0_owht_nohad_clip` can be traced back to
`LLAMA_KV_Q2_0_OWHT=1`, `LLAMA_KV_NO_HADAMARD=1`, and the configured
`LLAMA_KV_CLIP_RATIO`.

A CPU-only smoke test can verify the PPL pipeline on machines where CUDA/NVML is
blocked:

```bash
./scripts/make_ppl_smoke_corpus.sh /tmp/oscar_kv_ppl_smoke_corpus.txt

CORPUS=/tmp/oscar_kv_ppl_smoke_corpus.txt \
MODELS=granite:checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
CONTEXTS=short:64 \
KV_PAIRS=f16/f16,q8_0/q2_0 \
CHUNKS=1 \
N_GPU_LAYERS=0 \
MEASURE_VRAM=0 \
DRY_RUN=0 ACK_PPL_MATRIX=1 \
./scripts/run_kv_ppl_matrix.sh
```

Latest reproducible CPU/no-VRAM smoke:

| model | context | KV | PPL | ratio vs F16 |
|---|---:|---|---:|---:|
| granite | 64 | f16/f16 | 10.0976 | 100.00% |
| granite | 64 | q8_0/q2_0 | 124.4064 | 1232.04% |

Input: `/tmp/oscar_kv_ppl_smoke_corpus.txt`, 147824 bytes,
SHA256 `b6bc40acacb0716cc0884d5ad8873c65188cc0e880923d9fcc8b5064966f752c`.
This smoke test only checks execution, parsing, and coarse regression behavior.
Its PPL values are not a final quality claim because repository-derived text and
tiny contexts are not stable benchmark conditions.

The PPL gate catches the smoke ablation as a regression under a strict threshold:

```bash
./scripts/check_kv_ppl_gate.py /tmp/kv_ppl_repro_smoke/summary.csv --max-ratio 1.05
# fails: q8_0/q2_0 ratio is 12.3204
```

## Remaining Issues

- CUDA `q2_0` `SET_ROWS` currently mirrors centroid packing but does not
  use the full CPU-side head-level OWHT transform by default. OWHT-aware CUDA
  helper code is staged behind `LLAMA_KV_Q2_0_OWHT=1`. Under
  `LLAMA_KV_NO_HADAMARD=1`, the staged writer remains compatible with the
  generic direct scalar reader, so the remaining OSCAR INT2 quality drift should
  be treated as a quantization/attention-accuracy issue rather than a known
  q2 reader mismatch. Hadamard-applied OWHT writes are different: they still
  require a matching inverse-OWHT reader for both K and V inside CUDA vector
  attention. A naive attempt to add a second OWHT-enabled template branch inside
  the generic vector FA path made the `q2_0` vector instance too heavy to build
  reliably in this environment, so the next implementation should be a dedicated
  q2_0 OWHT vector kernel that decodes each K/V row or group once into
  shared/register storage instead of scalar-reconstructing the Hadamard group for
  every accessed element.
- Metal centroid reconstruction is now aligned with CPU/CUDA, but Metal still
  does not implement the CPU q2_0 inverse OWHT path in these FA helpers.
- The fused `q2_0` + F16 HP CUDA kernel is scalar and simple. It validates the
  path and avoids the expensive concat softmax fallback, but it is not yet a
  highly optimized warp/tile implementation. The gated OWHT branch now avoids
  repeated per-scalar inverse OWHT inside a row, but it still uses local float
  decode buffers and needs runtime PPL/speed validation before becoming default.
- CUDA staged q2_0 writer now mirrors the CPU `LLAMA_KV_NO_HADAMARD` and
  `LLAMA_KV_CLIP_RATIO` gates, but this is still a staged path. Plain CUDA
  vector FA and Metal FA remain on direct centroid decode unless their matched
  OWHT reader path is implemented.
- `q2_0` helps total VRAM only when KV cache is large relative to weights and
  allocator overhead. Small prompts and larger models can show little total
  peak reduction even though KV storage itself is smaller.
- The current benchmark rows are throughput/VRAM checks. PPL validation is now
  scripted, but still needs to be run on a fixed corpus for both local models
  after the CUDA OWHT path is implemented. OSCAR-style INT2 accuracy claims
  still need rot-kv calibrated models and task/perplexity evaluation.

## Reference Direction

The implementation direction matches the practical lesson from open-source KV
compression work, but the current local results also show why this has to be
implemented as an attention-kernel problem rather than only a storage-format
change.

Useful external reference points:

- KIVI shows that 2-bit KV can be practical, but its design does not treat K and
  V as identical tensors: it uses asymmetric quantization and different
  grouping choices for key-cache and value-cache tensors. That supports keeping
  the local scripts ready for separate `--cache-type-k` / `--cache-type-v`
  experiments instead of assuming one KV type is always optimal for both.
- Hugging Face's quantized-cache implementation keeps a residual high-precision
  cache before quantizing older tokens. That matches the local `q2_0_hp`
  direction: sink/recent F16 cache is a quality and latency tool, not a pure
  memory-minimum mode. It also explains why short contexts can lose speed
  without saving much total VRAM.
- llama.cpp already exposes quantized KV cache types at the CLI level, including
  separate K/V cache type flags. This reinforces the naming choice to keep the
  public type as `q2_0` and make backend differences explicit through kernel
  implementation and gated environment options, not through CUDA-only type
  names.
- Rotation-based KV work such as RotateKV points in the same direction as the
  OSCAR q2_0 OWHT path: low-bit quality depends on the transform being paired
  correctly with the reader. A writer-only transform is not acceptable because
  attention would read a different representation from the one that was
  quantized.

Concrete implementation constraints from those references:

- Do not expand the whole low-bit KV cache to F16/F32 before attention. That
  would erase the memory advantage and add bandwidth pressure.
- Decode `q2_0` K/V at row or group granularity inside CUDA attention, reuse the
  decoded values for the dot/value pass, and avoid per-scalar inverse OWHT.
- Keep direct `q2_0`, `q2_0_hp`, and OWHT-enabled `q2_0` as separately measured
  modes. They optimize different tradeoffs.
- Treat PPL as the gate for OWHT/default changes. A faster CUDA path is not
  enough if the K/V transform semantics drift from CPU or if quality regresses
  beyond the previous `q2_0` baseline.
- Add asymmetric K/V experiments only after the paired CUDA path is stable; the
  current full-matrix scripts intentionally measure same-type K/V first so the
  effect is easy to interpret.
