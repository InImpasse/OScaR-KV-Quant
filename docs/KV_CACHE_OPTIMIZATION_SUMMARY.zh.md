# KV Cache 优化阶段总结

## 结论

当前版本已经把 CUDA `q2_0` KV cache 路径跑通，并补齐了测量脚本、理论 KV
估算、PPL 验证脚本和 GPU 空闲保护。`q2_0` 对 **KV 本身** 的压缩很明显：
理论上约为 F16 KV 的 18.8%。但它对 **总显存峰值** 是否明显，取决于 KV cache
在总显存里占比有多大。

从已有 RTX 5050 Laptop 矩阵看：

- Granite 1B：中长文本下总显存下降明显，且 generation 速度基本不降。
- Gemma E2B：总峰值主要被模型权重和 runtime/allocator 占用主导，`q2_0`
  的 KV 节省没有充分反映到总峰值上。
- Prompt processing 对 `q2_0` 明显变慢，这是当前 CUDA 路径的主要性能问题。
- `q2_0_hp` 保留 sink/recent F16 高精度缓存，适合精度保护，不是短中上下文的
  最小显存模式。

## 两模型短中长结果

数据来源：
`runs/q2hp_bench_matrix_20260604T164856Z/kv_matrix_combined.md`

### Granite 1B

| 长度 | KV | 总峰值节省 | 理论 KV 节省 | pp 速度 | tg 速度 |
|---|---|---:|---:|---:|---:|
| short 512 | q2_0 | 36 MiB | 32.5 MiB | 78.2% | 106.8% |
| medium 2048 | q2_0 | 126 MiB | 130.0 MiB | 41.3% | 105.9% |
| long 4096 | q2_0 | 266 MiB | 260.0 MiB | 24.9% | 101.8% |

判断：Granite 的 `q2_0` 显存优化是明显的，尤其 2048/4096 tokens。测量节省与理论
KV 节省基本一致，说明总峰值在这些场景里能反映 KV 压缩效果。速度方面，prompt
processing 明显变慢；generation 与 F16 接近，部分行略高。

### Gemma E2B

| 长度 | KV | 总峰值节省 | 理论 KV 节省 | pp 速度 | tg 速度 |
|---|---|---:|---:|---:|---:|
| short 512 | q2_0 | 30 MiB | 28.4 MiB | 64.7% | 67.1% |
| medium 2048 | q2_0 | 44 MiB | 113.8 MiB | 45.2% | 64.7% |
| long 4096 | q2_0 | 4 MiB | 227.5 MiB | 1.9% | 58.3% |

判断：Gemma 的理论 KV 节省存在，但总峰值改善不明显，尤其 long 只看到 4 MiB。
这说明该 run 的总峰值被非 KV 部分主导，或者 allocator/执行路径峰值掩盖了 KV
节省。速度方面，Gemma 的 `q2_0` generation 也明显慢于 F16，说明当前 kernel mix
对 Gemma 形状不友好。

## q8/q4/q2 理论显存比例

两模型的 KV cache 理论存储比例一致：

| KV 类型 | 相对 F16 KV |
|---|---:|
| q8_0 | 53.1% |
| q4_0 | 28.1% |
| q2_0 | 18.8% |

也就是说，`q2_0` 对 KV 本身的压缩是确定的；不确定的是这个压缩能否显著降低
**总峰值显存**。

## 当前优化进展

- CUDA `SET_ROWS` 支持写入 `q2_0` KV。
- CUDA vector flash attention 可以直接消费 `q2_0` K/V。
- CUDA 增加了 `q2_0` + F16 HP fused attention，避免 LP/HP concat fallback。
- CPU/CUDA/Metal 使用同一个公开存储类型 `q2_0`。
- CPU/CUDA/Metal 的 Lloyd-Max INT2 centroid/threshold 常量已统一到
  `ggml-common.h`。
- CUDA staged OWHT helper 已拆到 `q2_0-owht.cuh`，并通过
  `LLAMA_KV_Q2_0_OWHT=1` gated；旧 `LLAMA_CUDA_Q2_0_OWHT=1` 保留兼容。
- CUDA staged group helper 现在跟随 CPU 的 `LLAMA_KV_NO_HADAMARD=1` 语义：
  仍使用 group mean + Lloyd-Max packing，但在已存在校准旋转时跳过 Hadamard 阶段。
- CUDA staged writer 也补齐了 CPU 的 `LLAMA_KV_CLIP_RATIO` 百分位 clamp 语义；
  默认 ratio 为 0 时不改变行为。
- 测量脚本支持两模型、短中长、多 KV 类型、VRAM 采样、理论 KV join、PPL 矩阵、
  busy GPU guard 和 wait-for-idle。
- 测量脚本现在支持 `KV_PAIRS=K/V,...` 非对称 K/V cache 实验，例如
  `q8_0/q2_0`、`q2_0/q8_0`。默认 `KV_MODES` 仍保持 K/V 同类型，方便和已有矩阵
  直接比较。
- 测量脚本现在也支持 CUDA q2_0 staged mode 标签：
  `q2_0_owht`、`q2_0_owht_nohad`、`q2_0_owht_clip`、
  `q2_0_owht_nohad_clip`。这些只是实验标签，实际传给 llama.cpp 的 cache type
  仍是 `q2_0`，用于区分 CUDA staged writer/reader 的环境变量组合。
- benchmark summarizer 在 join 理论 KV 存储时，会把 `q2_0_owht*` 当作与
  `q2_0` 存储等价处理，因此已有的 `q2_0` 理论 CSV 可以继续复用。
- PPL 脚本会在 `config.txt` 记录 `corpus_bytes` 和 `corpus_sha256`，并新增
  `scripts/make_ppl_smoke_corpus.sh` 生成固定本地 smoke 语料，避免临时文本导致
  PPL smoke 不可复现。
- 新增 `scripts/check_kv_ppl_gate.py`，可对 `summarize_kv_ppl.py` 输出的 CSV 做
  F16 baseline 相对 PPL 门槛检查。CUDA OWHT 或 q2_0 reader/writer 默认化前，应先
  通过这个 gate。
- 新增 `scripts/check_kv_bench_gate.py`，可对 benchmark summary CSV 做显存节省、
  速度 ratio、measured/theory saved 等门槛检查。
- benchmark/PPL 每行 summary 和 CSV 现在记录实际 `cache_type_k/v` 以及
  `LLAMA_KV_HP_*`、`LLAMA_KV_Q2_0_OWHT`、`LLAMA_KV_NO_HADAMARD`、
  `LLAMA_KV_CLIP_RATIO`，方便追溯 staged q2_0 结果到底用了哪组环境变量。
- busy GPU guard 现在会输出并记录 `guard_baseline_mib` 和 compute-process snapshot。
  当前机器仍显示 baseline 4425 MiB，且有 PID 312710 占用/残留，因此 busy smoke
  只能证明路径可执行，不能作为正式显存结论。

## 主要问题

- CUDA/Metal 默认 FA 路径仍是 direct centroid decode；CPU 默认 `q2_0` 是
  128-dim group OWHT 语义。这个语义差异仍未完全消除。
- 通用 CUDA vector FA 里粗暴加入 OWHT 模板分支会让 `q2_0` 实例过重，构建不可靠。
  下一步应做专用 q2_0 OWHT kernel，而不是把 heavy path 塞进通用模板。
- Prompt processing 慢，尤其 medium/long 的 q2_0 行。
- Gemma long 的 q2_0 prompt processing 极慢，需要单独排查 kernel path、allocator
  峰值和 FA 形状。
- `q2_0_hp` 的理论 HP F16 side cache 成本在现有 peak 采样里没有充分暴露，不能
  宣称它和 plain `q2_0` 显存等价。
- PPL/精度矩阵已脚本化，但还缺正式固定语料、两模型、多 KV 类型的 GPU/CPU 质量结果。
- 当前新增的 repository-derived PPL smoke 只能验证执行和解析，不能替代 WikiText 等
  标准语料上的正式质量结论。

## 参考开源/论文后的约束

参考 KIVI、Hugging Face quantized KV cache、llama.cpp quantized KV cache 和
RotateKV 后，下一步不应只改存储格式，而要把低比特 KV 当作 attention kernel 问题：

- K 和 V 不一定适合同一种量化粒度。当前矩阵先测 same-type K/V 是为了看清基础效果；
  现在可以用 `KV_PAIRS` 加 `cache-type-k` / `cache-type-v` 非对称实验。
- sink/recent F16 cache 有意义，但它是精度和延迟保护，不是短中上下文的最小显存方案。
- CUDA 不能先把整段 KV 展开成 F16/F32 再 attention；否则显存和带宽优势都会被抵消。
- q2_0 OWHT 必须 writer 和 reader 成对实现。只让 `SET_ROWS` 写 OWHT，而 FA 仍 direct
  decode，会直接破坏语义。
- PPL 必须作为默认化门槛。速度变快但困惑度明显回退，不算有效优化。

## 下一步

1. 等 GPU 空闲后跑正式矩阵：

```bash
WAIT_FOR_IDLE_GPU=1 \
GPU_IDLE_TIMEOUT_SEC=3600 \
GPU_IDLE_POLL_SEC=10 \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

2. 对矩阵结果执行 benchmark gate。现有矩阵中 Granite long `q2_0` 可通过下面门槛，
   Gemma long `q2_0` 会失败，正好对应当前结论：

```bash
./scripts/check_kv_bench_gate.py \
  runs/q2hp_bench_matrix_20260604T164856Z/kv_matrix_combined.csv \
  --model granite \
  --length long \
  --kv q2_0 \
  --min-delta-saved-mib 200 \
  --min-tg-ratio 0.95 \
  --min-measured-over-theory 0.8 \
  --fail-empty
```

3. 用固定语料跑 PPL：

```bash
CORPUS=/path/to/wiki.test.raw \
WAIT_FOR_IDLE_GPU=1 \
GPU_IDLE_TIMEOUT_SEC=3600 \
GPU_IDLE_POLL_SEC=10 \
DRY_RUN=0 ACK_PPL_MATRIX=1 \
./scripts/run_kv_ppl_matrix.sh
```

4. 本地 smoke 可以先用固定仓库语料验证 PPL 管线：

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

最近一次 smoke：Granite context 64，`f16/f16` PPL 10.0976，`q8_0/q2_0` PPL
124.4064。该结果只说明低比特路径和汇总可运行，不是正式质量结论。

5. 对 PPL summary 执行质量 gate：

```bash
./scripts/summarize_kv_ppl.py runs/kv_ppl_<UTC>

./scripts/check_kv_ppl_gate.py \
  runs/kv_ppl_<UTC>/kv_ppl_summary.csv \
  --max-ratio 1.05
```

在 smoke 结果上，`q8_0/q2_0` 会因 ratio 12.3204 大于 1.05 被拦截，说明 gate 能
捕捉明显 PPL 退化。

6. 跑 K/V 非对称补充矩阵，判断 K 或 V 哪一侧更适合低比特：

```bash
KV_PAIRS=f16/f16,q8_0/q2_0,q2_0/q8_0,q2_0/q2_0 \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

7. 跑 CUDA q2_0 staged mode ablation，区分 direct / OWHT / no-Hadamard / clip：

```bash
KV_MODES=f16,q2_0,q2_0_owht,q2_0_owht_nohad_clip \
Q2_0_CLIP_RATIO=0.96 \
DRY_RUN=0 ACK_MATRIX_BENCH=1 \
./scripts/bench_kv_cache_matrix.sh
```

这些 mode 不新增公开 GGML 类型，仍保持 CPU/CUDA/Metal 共同使用 `q2_0` 命名。

8. CUDA 实现方向：单独实现 q2_0 OWHT-aware vector attention，按 row/group 解码一次
   K/V，再做 attention，避免 per-scalar 重建 Hadamard group。

9. 针对 Gemma long 慢路径做 profiling，先定位是 q2_0 K dot、V dequant、mask/FA
   fallback，还是 allocator/fit 行为导致。
