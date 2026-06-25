# Q2 / OSCAR Rotation Accuracy Triage

Date: 2026-06-14

## Summary

`oscar_turbo2_streamk` is not `oscar_int2`.

- `oscar_int2` means the rotated Granite GGUF plus exact llama.cpp KV cache
  `q2_0/q2_0`.
- `oscar_turbo2_streamk` means the rotated Granite GGUF plus experimental
  `turbo2/turbo2` KV cache and `LLAMA_TURBO_VEC_STREAM_K=1`.
- The 32k speed row for `oscar_turbo2_streamk` must not be used as evidence
  that exact OSCAR INT2 is complete.

Current evidence does not support redoing the baked OSCAR rotation in
llama.cpp:

- `oscar_bf16` behaves like `baseline_bf16` on short accuracy smoke tests.
- `oscar_int4` behaves like `plain_int4` and BF16 on the same smoke tests.
- The rotated GGUF contains 80 `blk.N.attn_{k,v}_rot.weight` tensors, all
  128x128, with orthogonality error around `1e-7`.

The failure is concentrated in exact `q2_0` KV cache, and it appears in both K
and V isolation tests.

## INT4 accuracy boundary

INT4 is a healthy q4 control in the current llama.cpp evidence, not the same
failure class as exact INT2. It is not bit-identical to BF16, but the measured
drift is small enough that it supports the rotation sanity check rather than
contradicting it:

| variant | final-logit cosine vs oscar BF16 | final-logit NMSE | top10 overlap | per-layer min cosine | per-layer max NMSE | top token |
|---|---:|---:|---:|---:|---:|---|
| baseline_bf16 | 0.99998478 | 3.066559e-05 | 1.0 | n/a | n/a | same as oscar BF16 |
| oscar_int4 | 0.99472354 | 0.010540852 | 0.9 | 0.94569057 | 0.10882205 | same as oscar BF16 |
| oscar_int2 | 0.87953131 | 0.24424452 | 0.4 | 0.27657855 | 1.8331577 | different from oscar BF16 |

So the current answer to "why are `oscar_int4` and `plain_int4` also poor?" is:
they are slightly lossy, as q4 KV should be, but the local evidence does not
show the same correctness failure as INT2. The useful comparison is not
"INT4 equals BF16"; it is "INT4 remains close to BF16 and far better than
exact q2_0".

The 50-sample CLI smoke also keeps both INT4 variants in the BF16 band:

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 11/50 | 16/50 | BF16 control |
| oscar_int4 | 9/50 | 16/50 | same order as BF16 |
| plain_int4 | 9/50 | 15/50 | same order as BF16 |
| oscar_int2 | 0/3 | 0/3 | short failure probe, not a full score |
| plain_int2 | 0/3 | 0/3 | short failure probe, not a full score |

Both direct prompt files also answer `4` for plain/oscar INT4, while plain INT2
does not. Guard: `scripts/check_int4_accuracy_boundary.py`.

## Runs

CUDA flash-attention smoke:

- Archive: `runs/q2_isolation_cuda_faon/`
- Command shape: 3 GPQA questions, `llama-completion`, `-ngl 999`, `-fa on`,
  4096 context.

| variant | KV | result |
|---|---|---|
| baseline_bf16 | bf16/bf16 | 1/3, normal `Answer: A` output |
| oscar_bf16 | bf16/bf16 | 1/3, normal `Answer: A` output |
| plain_int4 | q4_0/q4_0 | 1/3, normal `Answer: A` output |
| oscar_int4 | q4_0/q4_0 | 1/3, normal `Answer: A` output |
| plain_int2 | q2_0/q2_0 | 0/3, repeated `- \\hbar/2` style output |
| plain_int2_nohad | q2_0/q2_0 | 0/3, same as plain int2 |
| plain_kq2_vbf16 | q2_0/bf16 | 0/3, malformed/repeated output |
| plain_kbf16_vq2 | bf16/q2_0 | 0/3, malformed/repeated output |
| oscar_int2 | q2_0/q2_0 | 0/3, LaTeX/token garbage |
| oscar_kq2_vbf16 | q2_0/bf16 | 0/3, repeated `-\\hbar/2` style output |
| oscar_kbf16_vq2 | bf16/q2_0 | 0/3, LaTeX/token garbage |

Direct prompt smoke:

- Archive: `runs/q2_direct_prompt_current/`
- Prompt: `Question: What is 2 + 2? Answer with one number.`

| variant | output summary |
|---|---|
| baseline_bf16 | answers `4` |
| oscar_bf16 | answers `4` |
| plain_int4 | answers `4` |
| oscar_int4 | answers `4` |
| plain_int2 | answers `2` |
| plain_kq2_vbf16 | malformed `0, 1, e2...` |
| plain_kbf16_vq2 | answers `1` |
| oscar_int2 | token garbage |
| oscar_kq2_vbf16 | repeats `2 + 2 2 2...` |
| oscar_kbf16_vq2 | token garbage |

Turbo prompt smoke:

- Archive: `runs/turbo3_direct_prompt_current/`
- Non-Stream-K `turbo3/turbo3` answered `4` on the direct prompt.
- Stream-K turbo2/turbo3 variants emitted repeated tokens on the direct prompt.
- `oscar_turbo3` is now a first-class eval/32k harness case: rotated GGUF plus
  `turbo3/turbo3`, without Stream-K. It exists to separate rotation behavior
  from exact `q2_0` behavior on a 3-bit TurboQuant-style KV cache.

This means local Turbo3 is useful as a TurboQuant reference/probe, but
Stream-K Turbo results are not quality evidence for OSCAR INT2.

Turbo3 CLI smoke:

- Archive: `runs/oscar_turbo3_cli_smoke_current/`
- Command shape: 3 GPQA + 3 GSM8K samples, `llama-completion`, 4096 context,
  CUDA flash attention on.

| variant | GPQA | GSM8K | output shape |
|---|---:|---:|---|
| baseline_bf16 | 1/3 | 1/3 | normal explanations/answers |
| plain_int3 (`turbo3/turbo3`) | 1/3 | 0/3 | GPQA mostly short answers; GSM8K empty/repeated markdown |
| oscar_turbo3 (`turbo3/turbo3`) | 0/3 | 0/3 | repeated options/brackets/sentences |

This keeps Turbo3 in the repo as a format and speed probe, but it is not a
validated quality replacement for OSCAR INT2. The failure mode is different
from the exact `q2_0` investigation only in degree; it still shows that the
current llama.cpp Turbo3 path lacks the full graph-side transform/quality
semantics needed before it can be used as a correctness argument.

## Reference int2 format boundary

The external reference "int2" label should not be treated as identical to
llama.cpp `q2_0`; it is not llama.cpp `q2_0`.

- llama.cpp `q2_0` uses `block_q2_0`: 32-value blocks with `d`, `m`, and packed
  Lloyd-Max centroid codes.
- The external reference int2 calibration/runtime path models group quantization
  with asymmetric scale/zero metadata, normally group size 128, plus separate
  high-precision prefix/recent windows.
- `turbo2/turbo2` is a separate dedicated KV type: one 128-value block, one
  fp16 norm, and packed 2-bit centroid indices. It is closer to a dedicated
  low-bit KV route than `q2_0`, but it is not exact `q2_0` and should not be
  counted as completed `oscar_int2`.

This format boundary explains why optimizing exact `q2_0/q2_0` did not recover
the external reference behavior: the target quantizer and the llama.cpp cache
type are mathematically different, not just differently optimized kernels.

## Offline quantizer comparison

Archive: `runs/int2_quantizer_comparison_current/`

The Granite FP16 QKV calibration dump was used to compare ordinary dequantized
NMSE for:

- llama.cpp `q2_0` Lloyd-Max block quantization,
- llama.cpp `q2_0` after applying the baked OSCAR rotation and rotating the
  dequantized tensor back,
- external-reference-style group-128 asymmetric scale/zero int2,
- the same asymmetric int2 after OSCAR rotation,
- TurboQuant-style `turbo2/turbo2` and `turbo3/turbo3` after OSCAR rotation.

| kind | rows | q2+rot mean | asym+rot mean | turbo2+rot mean | turbo3+rot mean | turbo3rot/q2rot |
|---|---:|---:|---:|---:|---:|---:|
| Kcur | 160 | 0.174535 | 0.251748 | 0.119648 | 0.088081 | 0.505 |
| Vcur | 160 | 0.173426 | 0.247206 | 0.118244 | 0.086326 | 0.498 |

This is an important negative result: ordinary NMSE alone does not explain the
q2 generation failure. On this dump, rotation improves `q2_0`, and rotated
`q2_0` has lower ordinary NMSE than the asymmetric int2 approximation. Turbo3
rotated ordinary NMSE is better than rotated `q2_0` by about 2x on both K and V,
which makes Turbo3 a useful 3-bit quality reference. The remaining exact q2
failure therefore points back to runtime semantics, attention-sensitive error
metrics, writer/reader pairing, or graph-side low-bit behavior rather than a
simple "q2_0 has larger elementwise NMSE" explanation.

## Offline attention-output comparison

Archive: `runs/int2_attention_error_current/`

The same calibration dump was also evaluated with a small causal attention
simulation: exact Q, dequantized K/V, Granite GQA head mapping, and per-layer
causal softmax. This compares attention-output NMSE rather than elementwise K/V
NMSE.

| rows | q2+rot mean | q2 OSCAR rot mean | q2 OSCAR K-only | q2 OSCAR V-only | asym+rot mean | turbo2+rot mean | turbo3+rot mean | turbo3rot/q2rot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 1.081504 | 1.040346 | 1.423517 | 0.185450 | 3.035318 | 1.439363 | 0.961298 | 0.889 |

This again does not support redoing the baked rotation: rotation sharply reduces
the q2 attention-output error on this dump. The current OSCAR-style q2
semantics (rotated, group-128 no-Hadamard, split K/V clip) are slightly better
than plain rotated q2 in this clean simulation. Turbo3 is the better 3-bit
quality reference in this attention-output simulation, while Turbo2 remains a
strong speed/reference-format path rather than the best attention-quality
replacement. The K-only/V-only split also matches the runtime layer-drift
direction: K=q2 is much more damaging to attention outputs than V=q2 in this
probe (`1.423517` vs `0.185450` NMSE), which explains why
`oscar_kq2_vbf16` has larger per-layer attention drift than
`oscar_kbf16_vq2`. Since real `oscar_int2` still emits token garbage, the next
concrete debugging target should be actual runtime KQ/softmax behavior or
generation-chain amplification, not regenerating the rotation.

## No-cache KQ / softmax sanity dump

Archive: `runs/q2_kq_softmax_dump_current/`

A tiny non-FA `llama-debug` probe was run on the rotated Granite GGUF with the
same direct prompt. This example uses the no-cache attention graph: `kq` is
built directly from `Qcur/Kcur`, not from `cache_k_set_rows`. It is therefore a
sanity probe for the explicit non-FA KQ/softmax graph, not final evidence about
KV-cache q2 KQ.

| variant | tensor | pass | worst layer | min cosine vs BF16 | max NMSE vs BF16 |
|---|---|---|---:|---:|---:|
| oscar_kq2_vbf16 | kq | prefill | 39 | 0.68874273 | 0.55378230 |
| oscar_kq2_vbf16 | kq_soft_max | prefill | 38 | 0.93177718 | 0.14160787 |
| oscar_kq2_vbf16 | kqv_out | prefill | 25 | 0.99999291 | 0.00001525 |
| oscar_kq4_vbf16 | kq | prefill | 39 | 0.98578322 | 0.03090660 |
| oscar_kq4_vbf16 | kq_soft_max | prefill | 0 | 0.95891579 | 0.08504313 |
| oscar_kq4_vbf16 | kqv_out | prefill | 26 | 0.99999224 | 0.00001663 |

The probe is still useful because the dumped `kq` matches direct Python
`Qcur @ Kcur.T` reconstruction for BF16 and q4 controls (minimum best cosine
`0.99999322` and `0.98998634`). The q2 path is the exception in this sanity
probe (minimum best cosine `0.78002906`), so it is a useful warning sign about
the low-bit graph path, but because `llama-debug` bypasses the KV-cache
attention graph it must not be used to claim that q2 cache KQ itself has been
proven correct or incorrect.

The same dump also includes `cache_k_set_rows`; analyzing those rows shows that
the q2 cache writer still matches the Python q2 writer to cache precision:
`max runtime-vs-python NMSE = 5.65e-08`, `max_abs_diff = 0.0219`. With
`LLAMA_KV_NO_HADAMARD=1`, the direct scalar reader is compatible with this
staged q2 layout: it reads `m + d * centroid(code)`, which is exactly the
no-Hadamard group-mean q2 reconstruction. The dequantized q2 cache rows still
have about `0.17-0.18` NMSE and cosine around `0.92` versus the source K rows on
representative layers, matching the offline q2 K quantization error. The
stronger q2 cache evidence therefore remains the cache reconstruction, direct
q2 cache KQ reconstruction, offline K-only attention split, FA layer drift, and
final-logit drift.

A direct q2 cache KQ reconstruction was added with
`scripts/analyze_q2_cache_kq_error.py`. It uses the runtime dumped
`Qcur_rot`, f32 `Kcur_rot`, and q2 `cache_k_set_rows` bytes, decodes cache K
through the same direct scalar q2 cache reader semantics, and compares
`Q @ K_f32.T` with `Q @ K_q2_cache.T`:

| layers | OWHT | no-Hadamard | min K cosine | max K NMSE | min KQ cosine | max KQ NMSE | worst KQ layer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 1 | 0.91122127 | 0.19055833 | 0.78010543 | 0.39160973 | 25 |

This is not a live CUDA FA kernel trace, but it is direct evidence from the
runtime q2 cache bytes that the K-side quantization loss is amplified by KQ
before softmax. That pushes the remaining OSCAR INT2 investigation toward K-side
format/precision/attention design rather than redoing the baked rotation.

## Rotation Evidence

The rotated GGUF was inspected with `gguf.GGUFReader`:

- 80 rotation tensors total.
- Names: `blk.N.attn_k_rot.weight` and `blk.N.attn_v_rot.weight`.
- Shape: 128x128.
- Example orthogonality max error: `3.5e-7` to `4.8e-7`.
- Diagonal mean of `R.T @ R`: `1.0`.
- The local Granite `.pt` rotations match the external reference checkout files
  byte-for-byte at tensor level.
- The baked GGUF tensors match the `.pt` rotations after transpose. This is the
  expected `export_rot_kv_gguf.py` layout: store `M^T` so
  `ggml_mul_mat(rot, K)` computes `K @ M`.
- Guard: `scripts/check_granite_rotation_alignment.py`.

Combined with `oscar_bf16` and `oscar_int4` smoke behavior, this strongly
suggests the baked rotation is loaded and applied coherently.

## Q2 Evidence

`test-quantize-fns -v` passes for `q2_0`, but the dot-product error is very
large:

- `q4_0 dot product error`: about `0.001142`
- `q2_0 dot product error`: about `0.479930`

This is large enough to explain why exact `q2_0` KV can pass format-level tests
while still breaking generation.

Environment switches did not rescue q2:

- `LLAMA_ATTN_ROT_DISABLE=1`
- `LLAMA_KV_HAD_SIZE=32`
- `LLAMA_KV_Q2_0_OWHT=1`
- `LLAMA_KV_NO_HADAMARD=1`

The generic CUDA q2 KQ LUT path was also checked statically:

- `scripts/check_q2_cuda_static.py` now parses `Q2_0_FATTN_SIGN_LUT` and
  `Q2_0_FATTN_HIGH_LUT` and validates the dp4a reconstruction against direct
  Lloyd-Max centroid dot products for all packed q2 bytes and representative
  signed q8 lane values.
- This passed, so the ordinary q2 KQ LUT byte order/sign decomposition is not
  currently the suspected hard bug.

One real CUDA semantic mismatch was fixed while investigating:

- CUDA `q2_0` OWHT/group `set_rows` used to store the group mean only in the
  first 32-value block.
- CPU reference quantization and per-block GPU decode semantics require the
  group mean to be available in every block.
- `third_party/OSCAR/ggml/src/ggml-cuda/q2_0-owht.cuh` now replicates the mean
  to every block, and `ggml-common.h` documents that layout.
- Guard: `scripts/check_q2_owht_mean_semantics.py`.

This was a correctness/consistency fix, but not a quality rescue. With
`LLAMA_KV_Q2_0_OWHT=1` and HP 64/256, `runs/q2_owht_mean_fix_probe_current/`
still showed 0/3 GPQA and 0/3 GSM8K for both plain and OSCAR INT2.

Another reference-semantics mismatch was aligned after comparing the external
Granite eval scripts:

- CUDA q2 OWHT `set_rows` now supports K/V-specific clip ratios:
  `LLAMA_KV_CLIP_RATIO_K=0.96` and `LLAMA_KV_CLIP_RATIO_V=0.92`, with
  `LLAMA_KV_CLIP_RATIO` as fallback.
- The split is selected from the destination KV cache tensor name
  (`cache_k_lN` / `cache_v_lN`) and is guarded by
  `scripts/check_q2_owht_mean_semantics.py`.
- Short probe with OWHT + HP 64/256 still did not rescue exact q2:
  `runs/gpqa_gsm8k_cli_eval_current/summary.md` has `oscar_int2` 0/3 GPQA and
  0/3 GSM8K after the split-clip change. Raw `oscar_int2` GPQA output still
  repeats tokens such as `\\hbar\\hbar...`, so the remaining issue is deeper
  than the K/V clip ratio.

The llama.cpp harness now enables the staged q2 OWHT writer for OSCAR-style
exact q2 runs:

- `oscar_int2` is still the rotated GGUF plus `q2_0/q2_0`; it is not Turbo2.
- The 32k bench and GPQA/GSM8K HTTP eval harness compute
  `LLAMA_KV_Q2_0_OWHT=1` only for `q2_0/q2_0` with
  `LLAMA_KV_NO_HADAMARD=1`, so plain q2, Turbo, BF16, and INT4 remain separate
  controls.
- The CLI eval keeps base `oscar_env()` q2-neutral and uses `oscar_q2_env()`
  to opt exact q2 variants into `LLAMA_KV_Q2_0_OWHT=1`.
- Guard: `scripts/check_oscar_int2_owht_harness.py`.

Runtime cache dumps now expose both the `ggml_set_rows` output and its source
rows/index tensors. The current archive is `runs/q2_runtime_cache_dump_current/`:

| mode | rows | max runtime-vs-python NMSE | max abs diff | interpretation |
|---|---:|---:|---:|---|
| ordinary CUDA q2 writer | 80 | 0.16319786 | 11.246727 | not aligned with the current OSCAR q2 reference semantics |
| staged q2 OWHT writer, no-Hadamard | 80 | 5.1846389e-08 | 0.015579224 | writer matches the Python q2 writer to fp16/cache precision |

This proves the staged OWHT `set_rows` writer itself is coherent when enabled.
However, a non-interactive `llama-completion` direct prompt still separates the
controls from q2 failure. Archive: `runs/llama_completion_direct_smoke_current/`.

| variant | KV | output sketch |
|---|---|---|
| oscar_bf16 | bf16/bf16 | `4  Question: What is` |
| oscar_int4 | q4_0/q4_0 | `2 Question: What is` |
| oscar_int2 | q2_0/q2_0 | `same>> [end of text]` |
| oscar_kq2_vbf16 | q2_0/bf16 | `2. 2 2` |
| oscar_kbf16_vq2 | bf16/q2_0 | `.123 ... linese?` |

This is not a conversation-mode artifact; the command uses `llama-completion`,
`-no-cnv`, `--simple-io`, and `--no-display-prompt`. The quality bug therefore
remains open and points more strongly at exact q2 K/V attention quality than at
the baked rotation or at the cache writer. K-only q2 becomes repetitive, while
V-only q2 produces malformed text.

- Reader/writer pairing is now split into two cases:
  `LLAMA_KV_NO_HADAMARD=1` q2 writes are compatible with the generic direct
  scalar reader (`m + d * centroid(code)`), so that mode is a real q2 quantizer
  quality test rather than a known reader mismatch. Hadamard-applied OWHT writes
  still require a matching inverse-OWHT reader; the staged inverse-OWHT reader
  exists in the fused q2_0+F16 HP kernel, but not in the generic q2/q2 vector FA
  path. Therefore enabling Hadamard-applied OWHT writes alone cannot be treated
  as an exact q2 prefill quality fix.
  Guard: `scripts/check_q2_owht_reader_limits.py` and
  `scripts/check_q2_nohad_reader_compat.py`.

- Runtime q2 cache KQ reconstruction now shows that direct scalar q2 cache
  reader semantics are sufficient to reproduce the K-side KQ drift from cache
  bytes: min K cosine `0.91122127`, max K NMSE `0.19055833`, min KQ cosine
  `0.78010543`, max KQ NMSE `0.39160973`. Guard:
  `scripts/check_q2_cache_kq_error.py`.

FutureMLS `zhongzhu/llamacpp` recommends OSCAR INT2 with a high-precision
sink/recent buffer:

- `LLAMA_KV_HP_SINK=64`
- `LLAMA_KV_HP_RECENT=256`

This was tested locally as `runs/q2_hp_gpqa_probe_current/` on 10 GPQA and 10
GSM8K samples. The result separates INT4 from INT2 clearly:

| variant | GPQA | GSM8K | note |
|---|---:|---:|---|
| baseline_bf16 | 2/10 | 3/10 | weak model/prompt baseline |
| oscar_bf16 | 2/10 | 4/10 | rotation does not hurt BF16 |
| plain_int4 | 3/10 | 3/10 | same order as BF16 |
| oscar_int4 | 3/10 | 4/10 | same order as BF16/INT4 |
| plain_int2 + HP 64/256 | 0/10 | 0/10 | still malformed/wrong on longer prompts |
| oscar_int2 + HP 64/256 | 0/10 | 0/10 | still malformed/wrong on longer prompts |

On a very short direct prompt, HP 64/256 can mask q2 because the visible KV is
covered by the high-precision recent cache. This does not fix long-prompt q2
quality.

An additional full-context HP probe used `LLAMA_KV_HP_RECENT=4096`:

- Archive: `runs/q2_hp_fullctx_probe_current/`
- 3 GPQA + 3 GSM8K samples.
- plain/oscar INT2 remained 0/3 on both datasets.
- BF16 and INT4 remained normal on the same probe.

This is expected from the current llama.cpp graph: during prompt prefill, HP
sink/recent cache is written but attention still uses the normal q2 FA path;
the HP+LP attention graph is gated to generation-sized batches
(`ubatch.n_tokens <= 2*ubatch.n_seqs_unq`). Therefore HP recent cannot be
treated as a q2 prefill quality fix.

An experimental llama.cpp-only gate was added to test the mixed-window
hypothesis directly:

- Env: `LLAMA_KV_HP_PREFILL_ATTENTION=1`
- Effect: allows LP+HP joint attention to be built for prompt batches when HP
  cache exists, instead of only generation-sized batches.
- Probe: `runs/q2_hp_prefill_attention_probe_current/`
- Result: `oscar_int2` remained 0/3 GPQA and 0/3 GSM8K with
  `LLAMA_KV_Q2_0_OWHT=1`, HP sink/recent 64/256, split K/V clip, and HP prefill
  attention enabled. Raw output still contains token garbage. This means simply
  enabling the HP attention graph during prefill is not enough to recover exact
  q2 quality in the current llama.cpp path.

Flash-attention on/off boundary:

- Archive: `runs/q2_fa_onoff_cli_smoke_current/`
- FA-on smoke for `plain_int2` and `oscar_int2` completed normally and produced
  0/3 GPQA plus 0/3 GSM8K for both variants, with repeated/token-garbage raw
  output.
- FA-off cannot be used as a full `q2_0/q2_0` quality control in current
  llama.cpp. A direct prompt with `-fa off -ctk q2_0 -ctv q2_0` exits with:
  `V cache quantization requires flash_attn`.
- `-fa off -ctk q2_0 -ctv bf16` is allowed and returns, but on the direct
  `2 + 2` prompt it immediately emits end-of-text. `-fa off -ctk bf16 -ctv
  q2_0` is rejected by the same V-cache guard.

This means the q2 failure cannot currently be isolated by a full FA-off
`q2_0/q2_0` run. The supported full q2 path is the FA path, and K-only FA-off
is too narrow to prove generation quality.

No-Hadamard graph gate fix:

- Code: `third_party/OSCAR/src/llama-kv-cache.cpp`
- Guard: `scripts/check_kv_no_hadamard_graph_gate.py`
- Direct smoke archive: `runs/no_hadamard_graph_gate_direct_smoke_current/`

Before this fix, `LLAMA_KV_NO_HADAMARD=1` only disabled the staged q2 writer's
Hadamard transform. The KV graph still enabled llama.cpp's generic attention
Hadamard rotation for any quantized K/V cache type. That made q2 experiments
internally inconsistent: the graph rotated K/V with an extra Hadamard, while
the q2 OWHT writer was told to use no-Hadamard semantics.

The fix makes `LLAMA_KV_NO_HADAMARD=1` also disable the graph-side attention
Hadamard rotation. This does not disable or replace the OSCAR GGUF
`attn_k_rot`/`attn_v_rot` matrices; it only gates llama.cpp's extra generated
Hadamard helper.

Direct prompt status after the fix:

| variant | output summary | note |
|---|---|---|
| oscar_bf16 | `4 ...` | healthy |
| oscar_int4 | `4 [end of text]` | recovered healthy short answer |
| oscar_int2 | `[end of text]` | still incomplete |
| oscar_kbf16_vq2 | `[end of text]` | V=q2 still incomplete |
| oscar_kq2_vbf16 | repeated `2` | K=q2 still degraded |

Q2 writer/clip A/B after the graph gate:

- Archive: `runs/q2_writer_ab_current/`
- Guard: `scripts/check_q2_writer_ab.py`
- 3+3 task probes:
  - `runs/q2_writer_ab_cli_eval_owht_noclip_current/`
  - `runs/q2_writer_ab_cli_eval_plain_writer_current/`

| q2 writer mode | result_output cosine | NMSE | top10 overlap | direct prompt |
|---|---:|---:|---:|---|
| legacy Hadamard mismatch | 0.17959263 | 2.3698112 | 0.1 | empty |
| OWHT no clip, no-Hadamard | 0.90744819 | 0.18778942 | 0.5 | `Solution: 2 + 2 =` |
| OWHT split clip, no-Hadamard | 0.87953131 | 0.24424452 | 0.4 | `[end of text]` |
| plain q2 writer, no-Hadamard | 0.91940516 | 0.15804489 | 0.6 | `[end of text]` |

Split K/V clipping (`0.96/0.92`) is worse than no clipping in this llama.cpp
path. The harness now keeps exact OSCAR q2 no-Hadamard but does not apply split
clipping by default. This is a cleanup of a harmful experimental default, not a
quality fix: both no-clip q2 probes remained 0/3 GPQA and 0/3 GSM8K.

Top-token drift triage:

- Script: `scripts/analyze_logits_top_tokens.py`
- Guard: `scripts/check_q2_top_token_drift.py`
- Ignore-EOS smoke archive: `runs/q2_ignore_eos_special_smoke_current/`

The remaining short-prompt q2 failure is not explained by EOS being promoted to
the top of the distribution. In `runs/q2_logits_path_dump_current/`, EOS remains
far down the rank list for `oscar_int2` and `oscar_kbf16_vq2`:

| variant | BF16 top-token rank | EOS rank | note |
|---|---:|---:|---|
| oscar_bf16 | 1 | 19390 | reference |
| oscar_int4 | 1 | 23328 | preserves first token |
| oscar_kq2_vbf16 | 1 | 18135 | K=q2 preserves first token in this prompt |
| oscar_kbf16_vq2 | 3 | 42570 | V=q2 shifts top token |
| oscar_int2 | 3 | 36698 | q2/q2 shifts top token |

With `--ignore-eos --special`, BF16/INT4 still start with `4`, while q2 emits
plausible but wrong continuations (`Solution: 2...` or `What is the...`). This
means the remaining issue is top-token/margin drift in exact q2, not an
EOS-special-token bug.

Layer-drift triage:

- Script: `scripts/analyze_layer_drift.py`
- Guard: `scripts/check_layer_drift.py`
- Archive: `runs/q2_logits_path_dump_current/layer_drift_summary.md`

Per-layer attention output drift, using `oscar_bf16` as the reference:

| variant | worst layer | min attention cosine | max attention NMSE | note |
|---|---:|---:|---:|---|
| oscar_int4 | 36 | 0.94569057 | 0.10882205 | small drift |
| oscar_kbf16_vq2 | 34 | 0.73583299 | 0.61504442 | V=q2 drift is moderate per layer |
| oscar_kq2_vbf16 | 11 | 0.42694113 | 1.9693227 | K=q2 has larger attention-score drift |
| oscar_int2 | 24 | 0.27657855 | 1.8331577 | q2/q2 remains severe |

This explains why INT4 should not be treated as the same failure class as INT2.
It also separates two effects: K=q2 causes large attention-output drift in
specific mid/late layers, while V=q2 is more directly implicated in final
top-token drift. The remaining q2 work should inspect exact q2 KQ score
quality and V reconstruction/margins, not redo the already-aligned rotation
matrices.

Q2 cache reconstruction error:

- Script: `scripts/analyze_q2_cache_reconstruction_error.py`
- Guard: `scripts/check_q2_cache_reconstruction_error.py`
- Archives:
  - `runs/q2_runtime_cache_dump_current/on/cache_reconstruction_error.md`
  - `runs/q2_runtime_cache_dump_current/on_clip/cache_reconstruction_error.md`

The q2 runtime writer now matches the Python q2 writer to fp16/cache precision,
but the quantized reconstruction itself is still lossy. With no split clipping:

| kind | worst layer | max NMSE vs source | min cosine | max abs |
|---|---:|---:|---:|---:|
| K | 0 | 0.21188222 | 0.90190813 | 169.70132 |
| V | 0 | 0.39698064 | 0.78650698 | 36.405369 |

With split K/V clipping (`0.96/0.92`), reconstruction gets worse:

| kind | worst layer | max NMSE vs source | min cosine | max abs |
|---|---:|---:|---:|---:|
| K | 0 | 0.230115 | 0.89487755 | 181.63391 |
| V | 0 | 0.47173077 | 0.75635041 | 37.719505 |

This connects the final-logits observation to the cache format: V=q2 has a
higher reconstruction error than K=q2, and split clipping worsens both. The
writer is internally consistent; the remaining issue is that exact q2_0 is too
lossy for this rotated Granite KV path without an additional accuracy strategy.

Offline q2 quantizer sweep:

- Script: `scripts/sweep_q2_quantizer_reconstruction.py`
- Guard: `scripts/check_q2_quantizer_reconstruction_sweep.py`
- Archive: `runs/q2_quantizer_reconstruction_sweep_current/`

Using the same dumped K/V source tensors, offline reconstruction confirms that
OWHT group-128 no-clip is the best of the tested exact q2 writer modes:

| kind | mode | mean NMSE | max NMSE | mean cosine | min cosine |
|---|---|---:|---:|---:|---:|
| K | plain q2 | 0.21822959 | 0.24611911 | 0.89605755 | 0.87923768 |
| K | OWHT no clip | 0.19077631 | 0.21188374 | 0.91161074 | 0.90190591 |
| K | OWHT split clip | 0.20221017 | 0.23011139 | 0.90860218 | 0.89487708 |
| V | plain q2 | 0.22744315 | 0.436582 | 0.89009293 | 0.75748764 |
| V | OWHT no clip | 0.20044231 | 0.39698449 | 0.90559666 | 0.78650649 |
| V | OWHT split clip | 0.22634961 | 0.47173005 | 0.89836132 | 0.75634873 |

Therefore the current harness keeps OSCAR exact q2 on OWHT no-clip rather than
falling back to the plain q2 writer. This is still not a quality fix; it is the
least-bad exact q2 writer among the local candidates.

Final logits-path tensor triage after the no-Hadamard graph gate:

- Archive: `runs/q2_logits_path_dump_current/`
- Script: `scripts/summarize_q2_logits_path_dump.py`
- Guard: `scripts/check_q2_logits_path_dump.py`
- Prompt: `Question: What is 2 + 2? Answer with one number.`
- Tensors: final `result_norm`, final `result_output` logits, layer-39
  `__fattn__`, and layer-39 `kqv_out`.

`oscar_bf16` is the reference. The base BF16 GGUF and rotated BF16 GGUF match
closely at logits level, which is another rotation sanity check. INT4 also
stays close to BF16. Exact q2 improves substantially after the graph gate, but
still does not reach INT4/BF16 stability:

| variant | result_output cosine vs oscar BF16 | result_output NMSE | top10 overlap | note |
|---|---:|---:|---:|---|
| baseline_bf16 | 0.99998478 | 3.066559e-05 | 1.0 | base vs rotated BF16 match |
| oscar_int4 | 0.99472354 | 0.010540852 | 0.9 | INT4 is close to BF16 |
| oscar_kbf16_vq2 | 0.94602882 | 0.12562276 | 0.7 | V=q2 no longer dominates after the graph gate |
| oscar_kq2_vbf16 | 0.92498112 | 0.14659257 | 0.5 | K=q2 remains degraded |
| oscar_int2 | 0.87953131 | 0.24424452 | 0.4 | improved, but still materially worse than INT4 |

All dumped tensors are finite: there is no NaN/Inf explosion. The earlier
V=q2-dominated semantic drift was largely caused by the no-Hadamard graph gate
mismatch. After fixing that mismatch, `oscar_int2` is much closer to BF16 at the
logits level, but the direct prompt still fails, so exact q2 quality remains
unfixed.

## Current Decision

Do not redo OSCAR rotation yet. A real llama.cpp integration bug was found and
fixed in the graph-side Hadamard gate, and the remaining issue is now the exact
`q2_0` KV cache path/format quality under the OSCAR rotation, not evidence that
the GGUF rotation matrices themselves are wrong.

Next implementation work should be one of:

1. Fix exact `q2_0` KV semantics only if a concrete mismatch is found.
2. Prefer a TurboQuant-style dedicated KV type (`turbo2`/`turbo3`) with a
   complete graph-side WHT/inverse-WHT path and quality validation.
3. Keep `oscar_turbo2_streamk` as a separate Turbo reference and never count it
   as exact `oscar_int2`.
