# OSCAR-KV-Quant — Agent Context

> 本文件记录 q2 KV cache 性能优化结论与实验过程，供后续 agent 直接加载，避免重复探索。

## 项目焦点

- **模型**：Granite 4.0 1B BF16（`checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`）
- **目标**：q2_0/q2_0 KV cache flash-attn prefill 性能
- **核心代码**：`third_party/OSCAR/ggml/src/ggml-cuda/fattn-common.cuh`、`fattn-vec.cuh`、`fattn.cu`
- **基准命令**：`llama-bench -m <model> -p <512|2048|8192> -n 64 -r 1 -ngl 999 -fa 1 --cache-type-k q2_0 --cache-type-v q2_0`

## 当前 baseline（精确 LUT，2026-06-11）

| prompt | q2_0/q2_0 pp (tok/s) | q4_0/q4_0 pp (tok/s) |
|---:|---:|---:|
| 512 | ~1671 | — |
| 2048 | ~915 | — |
| 8192 | **310** | **3638** |

- decode（tg64）：q2 ~67 vs q4 ~80 tok/s
- **8K prefill 差距约 12x**；decode 已接近 q4
- **保留门槛**：任何改动 8K 不得低于 **310 tok/s**；小优化需 **+10%** 且 512/2K 不明显回退

## q2 KQ 实现现状

- 路径：`sign/high LUT + 2× dp4a + m*usum`（精确，在 `fattn-common.cuh`）
- q4 KQ 只需 1× `dp4a`，精确 q2 很难靠 vec 内层局部改动追上 q4

## 已验证失败路线（勿重复）

| 实验 | 结果 |
|---|---|
| `GGML_CUDA_Q2_FATTN_FAST`（单次 dp4a 近似） | 无收益，已移除 |
| `GGML_CUDA_Q2_FATTN_TILE_D128` / shared-K tile | 无收益，已移除 |
| `ncols_partial` | 无收益，已移除 |
| D=128 KQ 内联 + `dm[4][2]` 寄存器缓存 | **-59%**（335→136），已撤回 |
| V dequant `ne==4` fma 改写 | 同上，已撤回 |

## q2 Speed Strategy 结论（2026-06-11）

1. **Phase 1**：干净 LUT baseline 已恢复，实验分支已清除
2. **Phase 2**：profiling 受 WSL 环境限制（见下节），微基准作替代证据
3. **Phase 3**：小步精确优化未达 +10%，全部撤回
4. **Phase 4**：D=128 q2/q2 专用 tile kernel → **NO-GO**

详细数字：`runs/q2_speed_strategy_results_20260611.md`

## OSCAR INT2 / mixed BF16 HP + INT2 bulk 进展（2026-06-16）

命名边界：

- 只保留三类目标/对照：`baseline_bf16`、`baseline_int2`、`oscar_int2`。
- 不再使用或新增“oscar plain int2”这类混淆命名；OSCAR INT2 指 **BF16 HP + INT2 bulk** 的 llama.cpp 底层实现方向。
- 当前工作必须限定在 llama.cpp / ggml CUDA 路径内，不转向 sglang harness。

当前实现/验证状态：

- CUDA runtime 已恢复正常，`llama-bench --list-devices` 能看到 RTX 5050 Laptop GPU，后续测试在非 sandbox GPU 环境可跑。
- 新增/继续维护的 mixed 路径在 `third_party/OSCAR/ggml/src/ggml-cuda/fattn-vec.cuh`、`fattn.cu`：
  - `flash_attn_ext_mixed_oscar2_f16_vec`：OSCAR2 LP bulk + F16 HP，共用 online softmax。
  - HP KQ 已改成复用寄存器里的 scaled Q，避免每个 HP key 重复从 global 读 Q。
  - LP OSCAR2 KQ 增加 shared `Q * OSCAR2_K_centroid[8]` LUT，减少每个 key 重复 centroid 乘法。
  - `LLAMA_KV_MIXED_VEC_NCOLS=8` 可做 8-query tile A/B；默认仍为 ncols=4，因为 8-query tile 只在 pp512 略好，pp2048 基本持平。
- `third_party/OSCAR/ggml/src/ggml-cuda/fattn-q2-tile-mixed.cu` 修过 q2 tile softmax correctness：`l_q` 不再只有 lane0 正确累加；alpha 可复用，避免重复 `expf`。
- 已尝试并撤回：
  - mixed LP V 专用 dequant helper：无收益/负收益。
  - mask early-out：pp2048 负收益。
  - `LLAMA_KV_MIXED_VEC_MAIN_PARTS=2/4`：只有很小变化，不是主瓶颈。
  - mixed VKQ half2 累加：pp512/pp2048 变慢，已撤回，保留 float2 累加。
  - 非 fused fallback（禁用 fused mixed，走 LP/HP KQ concat + joint softmax + split V）：pp512 ~109、pp2048 ~45，与 fused KQ-LUT 版接近，不是突破方向。
  - fused mixed 改传 F16 mask：pp512/pp2048 变慢，已撤回，仍使用 F32 mask。
  - HP recent window 扫描（64/128/256/512，pp2048）：均为 ~43-50 tok/s，同一量级；HP 段长度不是主瓶颈。
  - 诊断性跳过 HP correction（仅 LP bulk，pp2048）：~42.7 tok/s，仍和正常 mixed 一样慢；已撤回诊断开关。
  - mixed LP KQ 改为直接调用 plain oscar2 vec-dot helper：pp512 ~103.8、pp2048 ~42.0，低于 shared LUT 版；已撤回。
  - `LLAMA_KV_MIXED_VEC_LP_FAST_ONLY` 诊断（从 mixed launcher 内尝试调用 plain oscar2/oscar2 vec，只跑 LP bulk）：F32→F16 mask 转换版 pp512 ~98.8、pp2048 ~42.8；复用 graph F16 mask 后 pp512 ~104.0、pp2048 ~44.7；均未回到 direct oscar2/oscar2 vec 速度，诊断代码已撤回。
  - direct oscar2/oscar2 sanity 复测：无 HP mixed graph 时 pp512 ~238、pp2048 ~233；打开 HP prefill graph 环境但不走 mixed vec 时 pp512 ~90。说明 graph/launcher/mask 形态会显著影响所谓“plain vec”复用，不可只在 mixed launcher 内临时转调。
  - Plan A backend split (`LLAMA_KV_MIXED_VEC_SPLIT=1`)：LP raw numerator/meta + HP combine 能编译、smoke 通过，但 pp512 ~107.7 且 peak ~7067 MiB，pp2048 ~44.0；raw buffer 显存大且无速度收益，不应作为默认。
  - Plan B graph split (`LLAMA_KV_MIXED_VEC_GRAPH_SPLIT=1`)：LP 先走真正 `ggml_flash_attn_ext` plain oscar2/oscar2 节点，再由 combine kernel 重算 LP max/sum 并合并 HP；能编译、smoke 通过，但 pp512 ~111.7、pp2048 ~45.4，仍没有速度收益。结论：只要 combine 仍需重算 LP KQ/meta，速度回不来。

小规模验证数字（Granite 4.0 1B，`checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`）：

| case | prompt | env / note | pp tok/s | status |
|---|---:|---|---:|---|
| q2 tile | 8192 | `LLAMA_KV_Q2_TILE_MAIN=1`, q2_0/q2_0 | ~340 | ok，无 CUDA 报错 |
| plain oscar2 vec sanity | 512 | oscar2/oscar2，无 HP mixed | ~391 | ok，仅作底层 sanity，不作为目标命名 |
| plain oscar2 vec sanity | 2048 | oscar2/oscar2，无 HP mixed | ~230 | ok，仅作底层 sanity，不作为目标命名 |
| oscar_int2 mixed | 512 | HP recent=256, mixed vec ncols=4 | ~114-117 | ok，较早期 ~108 小幅提升 |
| oscar_int2 mixed | 2048 | HP recent=256, mixed vec ncols=4 | ~44-45 | ok，但仍远慢于目标 |
| oscar_int2 non-fused fallback | 512 | `LLAMA_KV_HP_NO_FUSED_Q2=1`, HP recent=256 | ~109 | ok，略慢于 fused KQ-LUT |
| oscar_int2 non-fused fallback | 2048 | `LLAMA_KV_HP_NO_FUSED_Q2=1`, HP recent=256 | ~45 | ok，与 fused KQ-LUT 接近 |
| oscar_int2 LP-fast-only diag | 512 | 从 mixed launcher 内转调 oscar2/oscar2 vec，F16 mask | ~104 | ok，诊断无收益，已撤回 |
| oscar_int2 LP-fast-only diag | 2048 | 从 mixed launcher 内转调 oscar2/oscar2 vec，F16 mask | ~45 | ok，诊断无收益，已撤回 |
| oscar_int2 backend split diag | 512 | `LLAMA_KV_MIXED_VEC_SPLIT=1`, LP raw + HP combine | ~108 | ok，peak ~7067 MiB，无收益 |
| oscar_int2 backend split diag | 2048 | `LLAMA_KV_MIXED_VEC_SPLIT=1`, LP raw + HP combine | ~44 | ok，无收益 |
| oscar_int2 graph split diag | 512 | `LLAMA_KV_MIXED_VEC_GRAPH_SPLIT=1`, LP plain FA + recompute meta combine | ~112 | ok，peak ~7067 MiB，无收益 |
| oscar_int2 graph split diag | 2048 | `LLAMA_KV_MIXED_VEC_GRAPH_SPLIT=1`, LP plain FA + recompute meta combine | ~45 | ok，无收益 |
| HP-only fallback | 2048 | HP recent=4096 覆盖当前上下文 | ~3800 | ok，说明 BF16 HP 直接路径很快 |
| 2+2 smoke | small | mixed oscar2 + HP recent=256 | 输出 `The result of 2+2 is 4.` | ok，无 `2. 2. 2...` 退化 |

结论：

- 目前已经不是驱动/runtime 问题，也不是 HP-only BF16 路径问题；HP-only 很快。
- plain oscar2 vec sanity 明显快于 mixed oscar2+HP，说明主要瓶颈在 **mixed LP+HP 共享 online softmax 的结构成本**，而不是 OSCAR2 KV type 基础存取本身。
- LP KQ shared LUT 有小幅正收益，但 pp2048/更长 prompt 仍被 mixed 主循环结构限制；继续只调 q2 scalar、CUDA graph、mask 分支或 V scalar dequant 基本没有意义。
- 非 fused fallback 与 fused KQ-LUT 速度相近，说明问题不只是 fused kernel 内部几行实现，而是 mixed joint-softmax/LP+HP 双段完整工作量本身。
- HP recent 从 64 到 512 变化不导致数量级差异，说明主要慢点不在 HP correction 短段，而在 LP bulk 进入 mixed joint-softmax 后没有复用 plain oscar2 vec 的高效主循环。
- 仅在 mixed CUDA launcher 内“转调”plain vec 不等于真正复用 plain vec 主路径：LP-fast-only 诊断仍慢，应转向 graph/launcher 级重构，让 LP bulk 作为真正独立的 plain FA 节点产出 numerator/meta，再由小 HP correction/combine kernel 合并 online softmax 状态。
- 跳过 HP correction 后仍慢，进一步确认瓶颈就是 **LP bulk 的 mixed loop 本身**，而不是 BF16 HP recent 合并。
- Plan A/Plan B v1 都失败的关键原因：两者都没有拿到 plain FA 内部的 LP raw numerator/meta。Plan A 自己产 raw 会引入大 buffer 且不够快；Plan B 复用 normalized LP output 但为了精确合并仍要重算 LP KQ max/sum，抵消了 plain FA 复用收益。

下一步方向：

- 把 mixed 路径改得更接近 plain oscar2 vec 的主循环：LP INT2 bulk 先走高效 plain vec/tiled loop，再把 BF16 HP 作为短段 correction 合并进同一个 online softmax 状态。
- 下一次真正可突破的方向是扩展 plain `flash_attn_ext_vec` 产出 raw numerator/meta（或新增专用 raw mode），让 HP combine 不再重算 LP KQ；否则不要继续在 split/graph split v1 上做小修。
- 也就是继续在 llama.cpp 底层实现 INT2 KV support：`baseline_bf16` 对照、`baseline_int2` 对照、`oscar_int2` 目标；不要再扩展出额外的“plain/oscar”混合命名。
- 32K 长跑仍需等 8K/中等上下文 mixed 路径有数量级改善后再恢复；当前阶段优先 pp512/pp2048/pp8192 A/B 和 correctness smoke。

## OSCAR INT2 raw mixed 速度更新（2026-06-18）

- Granite rotated GGUF baking 已复核：`PYTHONPATH=third_party/OSCAR/gguf-py` 读取
  `checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf`，存在 80 个
  `blk.{i}.attn_{k,v}_rot.weight` tensor（40 层 K/V 各一份）。当前测试确实在 rot-kv baked 模型上跑。
- `LLAMA_KV_MIXED_VEC_RAW_LP_TILE2=1` helper 已修到支持 Granite `logit_softcap=50`；
  p512 debug 能看到 `mixed_raw: lp_tile2=1`，说明入口实际命中。当前 helper 仍是 generic
  `flash_attn_ext_vec` stub，不是新的 tile kernel，因此不应期待稳定加速；pp2048 default/tile2
  单次波动很大，作为入口 sanity 使用。
- 在 generic `flash_attn_ext_vec` 的 half2 V 路径中加入了 OSCAR2 专用 inline 4-value V decode：
  保持 half2 accumulator 和输出语义不变，只减少 `dequantize_V_oscar2` hot loop 中重复 helper/`m,d`
  读取。构建通过。LP-only pp8192 复测出现约 697-700 tok/s 的好点，但 full `oscar_int2`
  pp8192 仍约 590-650 tok/s，说明最终瓶颈仍在 full raw mixed LP+HP pipeline，而不是单个
  V dequant helper。
- 同场 8K 对照（Granite 1B，`-p 8192 -n 64 -r 1 -ngl 999 -fa 1`）：
  `baseline_bf16` 约 3887 tok/s；当前 full `oscar_int2` 约 598 tok/s（有波动到约 650）。
  因此目标仍未达成：显存方向已有优势，但速度仍远低于 BF16。
- `LLAMA_KV_MIXED_VEC_HP_MASK_SKIP` 在 8K 下没有稳定收益。默认已改为关闭，仅保留 env
  显式开启复现；不要把 mask early-out 当主优化方向。
- 尝试过 OSCAR2 KQ loop 重排：把同一个 K 的 `qs/rs/idx` 解包移到 query 循环外，
  用 `sum_j[ncols]` 同时累加多个 query。结果明显负收益（LP-only pp2048 ~1311，
  LP-only pp8192 ~478，full pp8192 ~428），已回退。原因大概率是寄存器压力/occupancy
  损失超过减少 K 解包的收益；不要重复这类 `sum_j[ncols]` 寄存器展开。
- WSL profiler 状态仍不完整：`ncu` 最小 CUDA smoke 报 `LibraryNotLoaded`；`nsys`
  能生成 report/sqlite，但没有 CUDA kernel 表，无法做 kernel-level 时间分解。当前只能继续用
  llama-bench A/B 和代码静态路径判断。
- HP recent 长度诊断：`LLAMA_KV_HP_RECENT=0/64/256` 下 full pp8192 均约 590-603 tok/s，
  而 LP-only recent=256 约 697 tok/s。full 掉速约 15%，但不随 HP recent 增长，
  说明不是 HP 扫描长度主导；继续缩 HP window 或 mask early-out 收益有限。
- 小幅保留优化：OSCAR2 generic vec KQ 的 `nthreads_KQ` 从 2 改为 1。pp8192
  LP-only 约 700.7 tok/s，full 约 646-656 tok/s，较此前 full 常见 590-650 略稳。
  这只是小优化，离 BF16 仍有数量级差距。
- 尝试改 env-gated dedicated LP kernel（`LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP=1`）
  为 half LUT + KQ subgroup=1，仍明显慢于 generic：default pp8192 LP/full 约
  697/652 tok/s，dedicated 约 449/427 tok/s；已回退。dedicated 慢点不是 KQ LUT
  精度/子组大小，而是整体结构（尤其 float2 V accumulator/组合写回）不如 generic vec。
- 小幅保留优化：HP combine warp 在 `lp_normalized && parallel_blocks == 1` 时直接从
  normalized LP output + single meta 初始化 numerator/denom，跳过通用 part 循环。pp8192
  full 三次约 650/651/653 tok/s，LP-only 约 702 tok/s；收益不大但稳定且语义等价。
- 尝试过 shared `oscar2_v_lut[8][8]`，让 V half2 decode 直接查 centroid pair，
  避免每次构造 `make_half2`。结果负收益：LP-only pp8192 ~629，full ~587，已回退。
  额外 shared 初始化/同步/访问成本高于热循环里直接 `oscar2_dequantize_4_v_h2`。
- 尝试把旧 env-gated dedicated LP kernel 的 VKQ accumulator 从 float2 改成 half2，
  仍明显慢：`LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP=1` 下 pp8192 LP/full 约 459/439
  tok/s，已回退。这进一步确认旧 dedicated 结构不值得继续修，真正专用 kernel 需要重写
  tile/online-softmax/V 累加组织，而不是在旧 kernel 上替换类型。
- 尝试把 `LLAMA_KV_MIXED_VEC_RAW_LP_TILE2=1` 从 generic stub 改成 warp-per-key
  OSCAR2 LP tile kernel。修正 `threadIdx.y` warp 索引后，真实性能明显负收益：
  pp2048 LP-only 约 623 tok/s（默认约 1492），pp8192 LP-only 约 261 tok/s
  （默认约 697）。该内核已回退到 generic stub；不要继续沿 warp-per-key 方案微调。
- HP combine 范围裁剪不可直接做：`get_k_hp()`/`get_v_hp()` 会把真实 HP 数量 pad 到
  至少 256、再按 256 对齐；当前 `sink=64,recent=256,total=320` 时 `K_hp->ne[1]`
  已等于有效总量 320，没有大段 padding 空槽可跳过。`LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=1`
  三次 pp8192 full 为约 598/603/671 tok/s，`-r3` 平均约 621，低于默认 `-r3`
  约 630，仍不默认。
- 尝试把 HP combine warp 的 F16 K/V 读取改成 half2/float2 分组，构建通过但 pp8192
  full `-r3` 平均约 619 tok/s（样本约 570/642/645），低于改前默认约 630，已回退。
  说明 HP combine 当前不是通过简单 half2 load/convert 重写就能回收的瓶颈。
- 尝试 env-gated `LLAMA_KV_MIXED_VEC_FAST_SOFTCAP=1`，在 raw mixed LP/HP softcap
  中用 `__tanhf` 近似替代 `tanhf`。构建通过，但 pp8192 full `-r3` 平均约 611 tok/s
  （样本约 595/643/595），低于默认 `-r3` 约 630，tg 也从约 64 掉到约 59；已回退。
  Granite softcap 不是当前可通过 fast math 近似拿回速度的主瓶颈。
- 尝试把 HP combine warp 的 `logit_softcap != 0` 判断改成编译期模板分支。
  构建通过，但 pp8192 full `-r3` 平均约 627 tok/s（样本约 651/646/584），
  与默认约 630 持平略低，tg 约 56 也低于默认；已回退。HP combine 的控制流
  小修不是当前突破点。
- 尝试把 generic raw OSCAR2 KQ shared centroid LUT 从 half 改为 float，减少热循环中的
  `__half2float`。构建通过，但 pp2048 full 单次约 1357 tok/s，pp8192 full `-r3`
  平均约 642 tok/s（样本约 669/585/672），LP-only pp8192 单次约 616 tok/s，
  明显低于默认 LP-only 约 700；已回退。KQ LUT 的 half 转换不是主瓶颈，
  shared footprint/occupancy 更敏感。
- 尝试显式跳过 centroid-LUT K 路径中看似无用的 F32 `Q_reg` 初始化，只保留后续
  `turbo_lut` 初始化。构建通过，但 pp2048 full 单次约 1139 tok/s，明显低于默认，
  已回退。该初始化大概率已被编译器优化或影响模板寄存器布局；不要重复。
- 下一步不应继续调 HP window/mask/CUDA graph/q2 scalar。需要真正把 LP bulk 做成专用
  OSCAR2 D=128 tiled 主循环，或进一步让 plain vec 产出可被 HP combine 直接消费的 raw/meta，
  以避免 full path 中 LP-only 贴近 700 但 HP combine 后回落到 600 左右。
- `LLAMA_KV_MIXED_VEC_RAW_TILE_OSCAR2=1` 诊断（2026-06-18）：把 `fattn-tile.cuh`
  的通用 tile scaffold 泛化到 `GGML_TYPE_OSCAR2_KV`，K/V 在 KV tile 内 decode 到 shared，
  并在 normalized LP single-part 下强制写 `lp_meta` 供 HP combine 使用。构建通过且路径命中。
  同批 LP-only pp2048 default/tile 约 `1336/1582 tok/s`，说明结构在中等上下文有正信号；
  但顺序 LP-only pp8192 default/tile 约 `706.8/554.3 tok/s`，长上下文明显负收益。
  结论：这版“通用 tile scaffold + OSCAR2 decode staging”不能默认，也不应作为 8K gate
  方向继续微调；若继续 tiled 路线，应写更窄的 OSCAR2 D=128 专用主循环，降低通用 tile
  的 shared/register/half2 decode 开销。一次 8K default/tile 并行误跑得到约 300 tok/s，
  只能证明无 CUDA crash，不能作为性能数字。
- OSCAR2 KQ 8-dim 解包尝试（2026-06-18）：在 generic raw LP 的
  `nthreads_KQ==1` 路径里把原本每 4 维解包改成每 8 维一次读取两个 `qs` 和一个 `rs`。
  构建通过，但 LP-only pp2048 从同阶段约 `1497 tok/s` 掉到约 `564 tok/s`，明显负收益，
  已撤回。不要重复这类扩大 unroll/一次处理 8 维的 KQ 解包小修；寄存器/指令调度损失
  大于减少 load/shift 的收益。
- HP combine 诊断（2026-06-18，恢复后基线）：full pp2048/pp8192 约 `1509/653 tok/s`，
  同阶段 pp2048 LP-only 约 `1909 tok/s`，说明 full path 仍有约 20% 固定开销。
  `LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=1` 本轮 pp2048 约 `1638` 有小正收益，但 pp8192
  掉到约 `594`，仍不能默认。`LLAMA_KV_MIXED_VEC_HP_COMBINE_NORM1=1` 专用 normalized
  single-part combine kernel pp2048/pp8192 约 `1535/639`，8K 负收益，已撤回。不要继续
  从 HP combine 分支/part-loop 小特化榨速度；后续应回到 LP bulk 主循环或更大结构变化。
- HP window 复核（2026-06-18）：pp8192 full `HP_RECENT=0/256/512/1024` 约
  `598/653/658/634 tok/s`。512 与 256 基本持平，1024 变慢；HP recent 增大不是速度突破，
  但 512 可作为后续质量/速度权衡点。速度主线仍应放在 LP bulk 主循环。
- K-side residual V0 收敛测试（2026-06-22）：新增 env-gated
  `LLAMA_KV_OSCAR2_K_RESIDUAL=1`，不新增 public KV type、不改 `block_oscar2_kv`
  大小和 GGUF。实现方式是 K cache 写入时保留 block mean 到现有 `m` 字段，并用
  `(x - mean) / sigma` 选择现有 OSCAR2 8-level K centroid；V cache 不变。为保证一致性，
  CUDA scalar/vector/staging K dequant 和 C reference quant 都补齐 K `m` decode。默认关闭时
  K `m=0`，回到旧路径。
- 同场 8K bench（Granite 1B rot-kv，RTX 5050 Laptop，非 sandbox GPU，
  `-p 8192 -n 64 -r 1 -ngl 999 -fa 1`）：
  `baseline_bf16` pp8192 `3609 tok/s`；`baseline_int2` q2_0/q2_0 `335 tok/s`；
  `oscar_int2` HP6144/tight-view residual off `1735 tok/s`；
  `oscar_int2_residual` `1703 tok/s`；`oscar_int4` q4_0/q4_0 `3698 tok/s`。
  2+2 smoke 输出包含 “answer is 4”，无 `2. 2. 2...` 退化。
- 结论：K mean/residual V0 没有带来速度收益，且 8K 只达到 BF16 的约 47%，低于
  80% gate，因此本轮不跑 GPQA/GSM8K。不要继续调 residual 常数或 HP/mask/V helper；
  若继续 INT2，应进入更明确的 K3/V2 fallback 或承认纯 INT2 路线无法同时满足
  BF16 速度/质量目标。`oscar_int4` 仍是当前成功速度路线。
- 下一阶段收敛验证（2026-06-23）：按“收敛交付 INT4，限定验证 K3/V2”执行。
  K3/V2 最小候选尝试复用现有 `GGML_TYPE_TURBO3_0` 作为 K、`q2_0` 作为 V，仅临时
  接入 `turbo3/q2_0` D=128 FA vec instance、bench/common parser 和非
  `FA_ALL_QUANTS` dispatch。构建通过，但 `llama-bench -p16 -n1 --cache-type-k turbo3
  --cache-type-v q2_0` 在非 sandbox GPU 下直接 exit 139，WSL dmesg 出现 dxg
  completion errors；p512 同样崩溃。该候选未进入速度 gate，已撤回 dispatch/build/parser
  接线，不再作为可交付路线。若未来重启 K3/V2，必须先做新的安全 kernel/布局审计，
  不能仅靠打开现有 turbo3/q2_0 组合。
- 同日 `oscar_int4` 交付复测归档：`runs/oscar_int4_delivery_32k_20260623/`。
  32K BF16 vs INT4：`baseline_bf16` KV pool `2560.0 MiB`、peak `6142 MiB`、
  pp `2586.6 tok/s`、tg `49.4 tok/s`；`oscar_int4` q4_0/q4_0 KV pool
  `720.0 MiB`、peak `4306 MiB`、pp `2576.8 tok/s`、tg `36.8 tok/s`。
  INT4 peak 比 BF16 少 `1836 MiB`，几乎等于 KV pool 少 `1840 MiB`；pp 为 BF16
  的约 `99.6%`，满足 32K 速度/显存交付门槛。质量证据沿用
  `runs/mixed_k_turbo3_quality_eval_10_current/`：BF16 GPQA/GSM8K `3/10, 4/10`，
  INT4 `4/10, 4/10`，在同一小样本 band。

## 代码清理状态（2026-06-17）

本轮清理目标是把当前 llama.cpp/OSCAR 路径从实验堆叠状态收敛到只服务
`baseline_bf16`、`baseline_int2`、`oscar_int2` 的可验证分支。清理基线已归档：

- `runs/cleanup_audit_20260617/top_status.txt`
- `runs/cleanup_audit_20260617/oscar_status.txt`
- `runs/cleanup_audit_20260617/oscar_diff_stat.txt`
- `runs/cleanup_audit_20260617/oscar_diff_name_status.txt`

已清理内容：

- 移除 q2 tile mixed 失败实验入口与未跟踪 CUDA 文件：
  `fattn-q2-tile-mixed.cu/.cuh`，并删除 `fattn.cu`、`fattn.cuh`、
  `fattn-q2_0-f16.cu` 中的 dispatch/include/declaration。
- 移除 `LLAMA_KV_MIXED_VEC_SPLIT` 与
  `LLAMA_KV_MIXED_VEC_GRAPH_SPLIT` 的默认/环境触发入口；graph split v1
  不再参与 llama.cpp graph 构建。
- `LLAMA_KV_MIXED_VEC_RAW_LP_LUT`、
  `LLAMA_KV_OSCAR2_V_ACCUM_PACK`、`LLAMA_KV_MIXED_VEC_LP_VTILE`、
  `LLAMA_KV_MIXED_VEC_RAW_LP_ONLY` 不再读取环境变量，避免负收益/诊断路径被
  bench 环境误触发。
- 撤销 `common/debug.cpp` 里的临时 SET_ROWS dump 逻辑。
- 移除用户可见 `turbo2/turbo3` KV cache 参数入口和默认 CUDA FA template
  实例；底层 `GGML_TYPE_TURBO2_0/TURBO3_0` enum/type table 暂时保留，避免改变
  `GGML_TYPE_OSCAR2_KV` 的类型编号和 GGUF 兼容性。
- 默认 CUDA FA template 实例只保留当前目标/对照所需的
  `f16/f16`、`bf16/bf16`、`q2_0/q2_0`、`q4_0/q4_0`、`q8_0/q8_0`、
  `oscar2_kv/oscar2_kv`。

静态检查结果：

- `rg -n "LP_VTILE|OSCAR2_V_ACCUM_PACK|RAW_LP_LUT|GRAPH_SPLIT|Q2_TILE|q2_tile|LLAMA_KV_MIXED_VEC_SPLIT|RAW_LP_ONLY" third_party/OSCAR/ggml/src/ggml-cuda third_party/OSCAR/src third_party/OSCAR/common third_party/OSCAR/tools`
  无命中。
- `common/arg.cpp` 与 `tools/llama-bench/llama-bench.cpp` 仍支持 `oscar2`
  cache type；不再暴露 `turbo2/turbo3` cache type。
- `rg -n "sglang|SGLang"` 在 OSCAR llama.cpp 源码路径无命中；仅历史 summary
  / AGENTS 记录仍会提到 SGLang 作为已放弃对照。

验证结果（Granite 4.0 1B rotated GGUF，RTX 5050 Laptop，CUDA 12.9）：

- 构建通过：
  `cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench`
- 2+2 smoke 通过：
  `--cache-type-k oscar2 --cache-type-v oscar2` 输出 `The answer is 4.`
- 清理后 `oscar_int2` raw mixed bench：
  - pp2048：约 `1196-1199 tok/s`，贴近 `1259 tok/s` 历史门槛的 95%。
  - pp8192：约 `430-432 tok/s`，低于历史 `474 tok/s` 的 95%。
- 显式 `LLAMA_KV_MIXED_VEC_NCOLS=8` 在当前环境为负收益：
  pp2048 约 `536 tok/s`；不要把 ncols=8 当作默认优化方向。

当前结论：

- 清理后的源码能 build、能 smoke，且失败实验开关已从编译/dispatch 路径移除。
- 8K 性能没有完全回到历史单次记录，后续若继续优化应先回补/确认 raw mixed
  默认路径性能，再做新的 kernel 改动。
- 顶层 docs/scripts/runs 中仍有历史 turbo/SGLang/混合精度记录；它们不在当前
  llama.cpp 编译路径内。若继续整理，优先移动到归档目录，不要删除
  `AGENTS.md`、当前 progress summary、32K summary。

## 32k llama.cpp harness 结论（2026-06-12）

## OSCAR INT2 raw mixed 速度更新（2026-06-17）

- 当前保留优化：`fattn-vec.cuh` 中 `GGML_TYPE_OSCAR2_KV` 的 V 累加并行度从 `nthreads_V_q / 4` 调为 `nthreads_V_q / 2`（D=128 时 8 lane → 16 lane）。
- 当前默认路径：raw mixed 回到 generic `flash_attn_ext_vec<OSCAR2_KV, OSCAR2_KV, raw>` LP 主循环；`flash_attn_ext_mixed_oscar2_lp_raw_vec` 只保留为显式诊断/实验路径（`LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP=1`）。
- 重要 correctness 修复：旧 dedicated LP 的 OSCAR2 KQ LUT helper 每个 subgroup lane 只覆盖了部分 D=128 维度，导致 pp8192 ~1048 tok/s 的 fast 数字不可信；helper 已修成按 `nthreads_KQ` 分片覆盖完整 D=128，但修复后 dedicated LP 反而慢于 generic raw（pp2048 ~1238、pp8192 ~418），因此不能默认。
- 已尝试 raw mixed HP combine 长上下文 mask early-skip。后续 2026-06-18 A/B 显示 8K 无稳定收益，默认已关闭；仅保留 `LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=0|1` 用于复现。
- 该改动只影响 OSCAR2 V 的 CUDA FA vec 路径；外部命名仍只使用 `baseline_bf16`、`baseline_int2`、`oscar_int2`。
- Granite 4.0 1B rot-kv，`LLAMA_KV_MIXED_VEC_RAW=1 LLAMA_KV_MIXED_VEC_MAIN=1 HP_SINK=64 HP_RECENT=256`：
  - 清理后旧默认：pp2048 ~1249 tok/s，pp8192 ~471 tok/s。
  - 16-lane V：pp2048 单跑 ~1524 tok/s，pp8192 单跑 ~663-673 tok/s。
  - 正确 generic raw + HP mask skip：pp2048 约 `1650-1670 tok/s`（2K 基本不回退），pp8192 强制 A/B 中 skip0 ~685、skip1 ~705 tok/s，8K gate 勉强过线但仍远低于 BF16。
  - 8K 显存对照仍成立：oscar_int2 peak ~3695 MiB，baseline_bf16 peak ~4187 MiB。显存已优于 BF16，但速度仍显著慢于 BF16，目标未完成。
- 已验证负收益，勿重复：
  - OSCAR2 V helper 手写展开（float fallback 和 half2 分支）无收益或变慢。
  - OSCAR2 V 32 lane：pp2048 ~718 tok/s，pp8192 ~547 tok/s，明显变慢。
  - OSCAR2 `V_rows_per_thread=2`：pp2048 ~1273 tok/s，pp8192 ~527 tok/s，明显变慢。
  - OSCAR2 KQ 4 lane：pp2048 ~1134 tok/s，pp8192 ~485 tok/s，明显变慢；KQ 保持 2 lane。
  - `RAW_PARTS=2`：pp2048 ~1150 tok/s，pp8192 ~414 tok/s，不能用 parts 并行补长上下文。
  - 去掉尾部 shared 初始化/guard 化规约：pp2048 可到 ~1601 tok/s，但 pp8192 掉到 ~544 tok/s，不能保留。
  - 临时忽略 raw mixed softcap：pp8192 ~675 tok/s，只比正确路径小幅提升，softcap 不是主要瓶颈；不能作为正确路径。
  - 全局把 `flash_attn_ext_vec` 改成 256 threads：编译失败，旧 `flash_attn_ext_mixed_oscar2_f16_vec<128,8>` shared memory 0xc800 > 0xc000；若要试更多 warps，必须做 OSCAR2 raw 专用实例，不能全局切 vec helper。
  - 在 raw mixed LP 上复用普通 FA 的 `flash_attn_mask_to_KV_max`：pp2048 掉到 ~476 tok/s，可能与 LP effective tail/mask shape 或预处理开销不匹配；已撤回。
  - raw OSCAR2 KV tile 64 key（block 仍 128 threads）：pp2048 ~1092 tok/s，浪费 warp 并变慢；已撤回。
  - 临时接回旧 `flash_attn_ext_mixed_oscar2_lp_raw_vec`：pp2048 ~649 tok/s，pp8192 ~213 tok/s，不能作为默认或基础。
  - 修正完整 D=128 KQ 后的 dedicated LP 默认：pp2048 ~1238、pp8192 ~418，慢于 generic raw；不要再把旧 pp8192 ~1048 数字当作有效目标基线。
  - HP combine `single_normalized_lp` 专用化（parallel_blocks=1、LP 已 normalized）：pp2048 ~1343、pp8192 ~549，明显负收益，已撤回。
  - `LLAMA_KV_MIXED_VEC_NCOLS=8` 在正确路径 + HP mask-skip 后仍为负收益：pp8192 ~498 tok/s；ncols=1 ~568、ncols=2 ~684，也不如 ncols=4；保持 ncols=4。
  - HP mask skip 分支改成 lane0 读 mask 再 warp broadcast：pp8192 ~690，慢于每 lane 直接读 mask 的 ~705；已撤回。
  - OSCAR2 V fast path 手写 `oscar2_dequantize_4_v_f2` 直接累加：LP-only pp8192 ~583，慢于通用模板 LP-only ~725；已撤回。
  - `LLAMA_KV_MIXED_VEC_RAW_OSCAR2_V_THREADS=8|32` 诊断（2026-06-18）：LP-only pp8192 与默认几乎持平（~673 vs ~674），但 pp2048 明显回退（~1450/1472 vs ~1835 同批默认）；V lane 粒度不是当前突破点，不应默认。
  - `LLAMA_KV_MIXED_VEC_RAW_CAUSAL_KVMAX=1` 诊断（2026-06-18）：LP-only pp8192 与默认持平（~674），pp2048 回退（~1443 vs ~1791 同批默认）；简单按 `ic0+ncols` 裁剪 LP `k_VKQ_max` 不是可用突破点，不应默认。
  - `LLAMA_KV_MIXED_VEC_RAW_NORM_LP=0` 或 `RAW_PARTS=auto`：pp2048 ~512 tok/s，不能作为主路径。
- plain `oscar2/oscar2` FA sanity 当前比 raw mixed 慢（pp2048 ~385 tok/s，pp8192 ~282 tok/s），后续不要再假设“复用 plain FA 主循环”一定更快；需要继续优化 raw mixed 当前快路径内部结构。

仅使用当前 repo 的 llama.cpp/OSCAR `llama-bench`，不使用非 llama.cpp harness。
汇总报告：

- `runs/llamacpp_32k_kv_matrix_current/combined.csv`
- `runs/llamacpp_32k_kv_matrix_current/combined.md`
- 原始 run 归档：`runs/llamacpp_32k_kv_matrix_current/raw/`
- `docs/LLAMACPP_32K_KV_TEST_PLAN.md`
- `docs/Q2_32K_OPTIMIZATION_NOTES.md`
- `docs/EXTERNAL_REFERENCE_COMPARISON.md`

无 GPU 一致性检查：

```bash
scripts/verify_llamacpp_32k_kv_no_gpu.sh
```

该检查默认不访问 GPU，连 `nvidia-smi` 快照也会跳过；若人工需要只读 idle 快照，显式设置 `CHECK_GPU_SNAPSHOT=1`。该检查包含 `scripts/check_q2_cuda_static.py`，会确认 q2 CUDA 仍是精确 LUT baseline，并阻止已失败实验宏/变量重新混入。
同时包含 `scripts/check_llamacpp_only.py`，用于保持当前 repo 与 OSCAR 子模块在 llama.cpp-only 路线上（跳过 build/raw/vendor 噪音）。
还包含 `scripts/audit_goal_status.py`；在 32k INT2 没有有效速度前，它必须报告 `overall_status=incomplete`。
当前目标状态归档：`runs/goal_status_current/`。

| variant | prompt | status | KV | KV pool MiB | peak MiB | pp tok/s | tg tok/s | note |
|---|---:|---|---|---:|---:|---:|---:|---|
| baseline_bf16 | 32768 | ok | bf16/bf16 | 2560.0 | 6160 | 2486.4 | 41.6 | final BF16 baseline |
| oscar_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2533.8 | 39.2 | peak drop ≈ KV drop |
| plain_int4 | 32768 | ok | q4_0/q4_0 | 720.0 | 4324 | 2265.0 | 41.0 | healthy |
| plain_int2 | 16384 | ok | q2_0/q2_0 | 240.0 | 3792 | 180.0 | 44.1 | 16k gate only |
| oscar_int2 | 16384 | ok | q2_0/q2_0 | 240.0 | 3796 | 183.7 | 28.0 | rotation does not change speed materially |
| oscar_int2 | 32768 | failed | q2_0/q2_0 | 480.0 | 4036 |  |  | 480s timeout, empty JSON |

结论：

- **INT4 可用**：32k BF16 → INT4 理论 KV pool 少 1840 MiB，实测 peak 少 1836 MiB，显存下降已接近 KV cache 下降值。
- **32k exact q2_0/q2_0 当前 NO-GO**：OSCAR INT2 480s timeout 后 JSON 为空，无有效速度。
- **OSCAR rotation 不是 q2 速度瓶颈**：16k plain/oscar int2 prefill 基本相同（180.0 vs 183.7 tok/s）。
- **CUDA graph 不太可能救 32k q2 prefill**：问题是长 prefill 中 q2 KQ/V 路径过重，不是小 kernel launch overhead。
- 后续 CUDA 优化方向转为参考 FutureMLS `zhongzhu/llamacpp` 的 Metal tiled mixed-FA 设计，而不是继续在当前 q2 vec 内层微调。设计记录：`docs/FUTUREMLS_Q2_CUDA_PORT_PLAN.md`。
- FutureMLS Metal 性能核心是 `kernel_flash_attn_mixed_mm_q2_0_f16_d128` / `mm_mixed_pass`：Q=8 query tile、C=64 KV tile、q2 K/V dequant 一次并跨 query tile 复用、LP q2 与 HP f16 共享在线 softmax。CUDA 方向应移植这个 tiled prefill 思想到 D=128 q2/q2，而不是启用本地 `fattn-q2_0-f16.cu` 的朴素逐 KV/逐 D kernel。
- `build-cuda` 已有 `GGML_CUDA_GRAPHS=ON`；harness 可用 `CUDA_GRAPHS_MODE=on|off|auto` 与 `CUDA_GRAPH_OPT=1` 做低风险 A/B；优先用 `scripts/cuda_graph_ab.sh`，它默认 dry-run 且拒绝 32k。
- 512-token `plain_int2` CUDA graph A/B 已跑：graph off 2039.0 pp tok/s，graph on+opt 2020.6 pp tok/s；无收益，不支持直接加码到 32k q2。摘要归档：`runs/cuda_graph_ab_512_current/`。
- 后续 32k q2/int2 run 必须同时设置 `ACK_HEAVY_32K=1`、`ACK_Q2_32K_NOGO=1` 和 `ACK_Q2_RAMP_GATE_HOLD=1`，且只能单 case、`GEN_TOKENS=1`、`REPETITIONS=1`；默认应转向代码/profiler 优化，不要重复长跑。
- `scripts/bench_32k_llamacpp_kv.sh` 默认 `DRY_RUN=1`；任何真实 32k run（包括 BF16/INT4）都必须显式 `DRY_RUN=0`，q2/int2 还需要上面的 ACK。
- WSL 刚崩过时，恢复真实测试建议同时设置 `CASE_TIMEOUT_SEC` 和 `MAX_PEAK_MIB`；`measure_vram.sh` 会在采样显存超过 `MAX_PEAK_MIB` 时终止子进程并在 summary 记录 `limit_triggered=1`。
- 多个单 case 连续恢复测试时建议设置 `POST_CASE_COOLDOWN_SEC=30`，让 harness 在 case 之间等待 GPU 显存/利用率回落到 baseline guard 以下。
- 恢复测试前先运行 `python3 scripts/report_recovery_readiness.py`，只读汇总当前 incomplete 原因、q2 ramp gate、GPU 快照和 dry-run/real-command 打印入口；默认不访问 GPU，可用 `--no-gpu` 显式跳过 snapshot。
- 可用 `python3 scripts/report_q2_ramp_gate.py` 只读查看当前 q2 阶梯状态；当前 32k q2 已有 NO-GO 失败记录时应显示 `recommendation=hold_32k_q2`，优先代码/profiler 而不是重跑 32k。
- 可用 `python3 scripts/print_32k_q2_ramp_commands.py` 只打印 512→2k→4k→8k→16k→32k q2 recovery ramp 命令；该 helper 默认 `DRY_RUN=1` 且不执行 benchmark，若要打印 512/2k/4k/8k/16k 的 `DRY_RUN=0` 命令必须同时传 `--real --ack-real`，32k q2 真命令还必须额外传 `--ack-32k-q2-real --ack-q2-ramp-gate-hold`。
- 可用 `python3 scripts/print_32k_matrix_commands.py` 只打印完整 32k llama.cpp 矩阵命令；q2/int2 会保持单 case，`plain_int3` 只输出 unsupported 说明，真实命令打印要求 `--real --ack-real`，其中 32k q2/int2 真命令还必须额外传 `--ack-32k-q2-real --ack-q2-ramp-gate-hold`。
- plain int3 不支持：此 llama.cpp 分支未暴露 3-bit KV cache type；`Q3_K` 是权重量化格式。`scripts/check_kv_cache_types.py` 会静态验证这一点。

## Profiling 环境（WSL2 + RTX 5050）

### 症状

| 工具 | 错误 / 现象 |
|---|---|
| `ncu` | `LibraryNotLoaded` — profiler 无法加载驱动侧性能计数库 |
| `nsys` | profile 成功但 report **不含 CUDA kernel data** |

- 最小 CUDA 程序（`/tmp/oscar_ncu_smoke`）同样失败 → **环境级问题**，非 llama-bench 特有
- CUPTI API 初始化 OK，但 Nsight 计数器/trace 不可用

### 根因

WSL2 下 GPU profiling 依赖 **Windows 主机**：

1. **GPU Performance Counters 未开放**（最常见）
2. **WSL stub 库与 Windows 驱动不同步**（`nvidia-smi` 报 592.15，stub 可能仍是 590.67）
3. **Toolkit/Driver CUDA 主次版本不一致**（驱动 CUDA 13.1，toolkit CUDA 12.9）

官方说明：[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) — profiler 在 WSL 仍为 preview，需 Windows 侧配置。

### 修复步骤（Windows 主机操作）

```
1. NVIDIA Control Panel → Desktop → Enable Developer Settings
2. Developer → Manage GPU Performance Counters
   → "Allow access to the GPU performance counter to all users"
3. PowerShell: wsl --shutdown
4. 重新打开 WSL，运行诊断：
   ./scripts/ncu_wsl_preflight.sh
```

可选版本对齐（驱动 CUDA 13.x 时）：

```bash
sudo apt install cuda-nsight-compute-13-1   # 或 nsight-compute-2025.4.1
export NCU_BIN=/usr/local/cuda-13.1/bin/ncu
```

参考：[ERR_NVGPUCTRPERM](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

### 诊断与 profiling 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/ncu_wsl_preflight.sh` | WSL profiler 环境诊断 + 修复指引 |
| `scripts/q2_profile.sh` | ncu/nsys 尝试 + segment microbench fallback |
| `scripts/q2_segment_bench.sh` | q2/q2 vs q4/q4 及混合 K/V 分段对比 |

```bash
# 诊断 profiler 环境
./scripts/ncu_wsl_preflight.sh
# 完整 profiling 流程（profiler 不可用则自动 fallback）
DRY_RUN=1 ./scripts/q2_profile.sh
# 仅微基准
DRY_RUN=1 ./scripts/q2_segment_bench.sh
```

`q2_profile.sh` / `q2_segment_bench.sh` 默认 `DRY_RUN=1` 且使用 `third_party/OSCAR/build-cuda/bin/llama-bench`；真实运行必须显式 `DRY_RUN=0`。
单次推理入口 `scripts/run_llamacpp.sh` 默认 `DRY_RUN=1`；真实运行必须设置 `DRY_RUN=0 ACK_RUN_LLAMA=1`。
旧入口 `scripts/bench_kv_cache.sh` 也默认 `DRY_RUN=1`；若要真实跑默认 32k/4096/512 强度，必须同时设置 `DRY_RUN=0 ACK_HEAVY_CONTEXT=1`。
矩阵入口 `scripts/bench_kv_cache_matrix.sh` 同样默认 `DRY_RUN=1`，真实运行必须设置 `DRY_RUN=0 ACK_MATRIX_BENCH=1`。
PPL 矩阵入口 `scripts/run_kv_ppl_matrix.sh` 默认 `DRY_RUN=1`，真实运行必须设置 `DRY_RUN=0 ACK_PPL_MATRIX=1`；dry-run 不检查 corpus、不跑 preflight、不读 GPU。
`scripts/check_q2_profile_safety.py` 会静态检查这些低风险默认值。
`scripts/check_legacy_bench_safety.py` 会检查旧 KV bench/matrix/PPL 入口没有绕过 dry-run/ACK 护栏。
`scripts/check_execution_entrypoints.py` 会扫描 `scripts/` 中可触发 llama.cpp/profiler/GPU 采样的入口，防止新增脚本绕过 dry-run/ACK 分类。
`scripts/check_build_defaults.py` 会确认构建/运行/bench 脚本默认指向 `third_party/OSCAR/build-cuda`，并静态检查 `CMakeCache.txt` 中 `GGML_CUDA=ON`、`GGML_CUDA_FA=ON`、`GGML_CUDA_GRAPHS=ON`、Release、sm_120/CUDA 12.9 以及 `llama-bench`/`llama-cli`/`llama-perplexity` 可执行文件存在。
`scripts/report_q2_cuda_path.py` 只读 CUDA 源码并输出 q2/q4 KQ/V 静态路径事实（q2 KQ 3×dp4a、q4 KQ 1×dp4a、q2 V scalar decode、dispatch 映射）和函数体 sha256 fingerprint，用于无 GPU 地确认后续内核改动没有误判路径，也避免在 q2 代码未变时重复 32k 长跑。
当前 q2 CUDA 静态路径归档在 `runs/q2_cuda_path_current/`；`scripts/verify_llamacpp_32k_kv_no_gpu.sh` 会校验 SHA，并比较当前临时报 告与归档 CSV，若 q2 CUDA 源码改动后未刷新归档会失败。

### 微基准替代结论

- q2/q2 主路径慢，混合 K/V（q2/f16、f16/q2）极慢，**不可用于定位 q2/q2 子段瓶颈**
- 无 kernel 级 counter 时，足以否定继续在 vec 内层小修补
- 若需指令级证据：必须在 **原生 Linux + 匹配 Nsight 驱动** 或修复 WSL profiler 后重跑 `ncu`

## 后续优化方向（超出当前范围）

1. 修复 WSL profiler → `ncu` 采 `flash_attn_ext_vec<128,4,q2_0,q2_0>` 瓶颈
2. 若 profiling 支持且预计 2x+：独立 D=128 q2/q2 tile kernel（KQ+softmax+V 全链路），默认关闭
3. 专用 kernel 成功标准：8K ≥ 700 tok/s；否则保持 LUT baseline
4. 追平 q4：需近似路径或新 KV 格式，非精确 q2_0 短期目标

## 硬件 / 环境

- GPU：NVIDIA GeForce RTX 5050 Laptop（sm_120，CC 12.0）
- OS：WSL2 Ubuntu 22.04
- Driver：592.15（CUDA max 13.1）
- Toolkit：CUDA 12.9，`ncu` 2025.2.1

## 历史 runs 目录

- `runs/q2_lut_restored_verify_20260611T054553Z/` — baseline 验证
- `runs/q2_segments_20260611T054712Z/` — segment microbench
- `runs/q2_profile_retry_20260611T054714Z/` — ncu/nsys 失败记录

## OSCAR INT2 当前状态补充（2026-06-18）

- 修复一次临时诊断残留：不要在 CUDA device/global kernel 内调用 `getenv()`。
  `LLAMA_KV_MIXED_VEC_RAW_NO_K_LUT` 曾被直接放进 `flash_attn_ext_vec`
  device 路径，导致 `fattn-vec.cuh` 编译报
  `calling a __host__ function("getenv") from a __global__ function`。该诊断已移除，
  OSCAR2 KQ 恢复为 shared `Q * centroid` LUT 路径。以后若需要这类 A/B，必须做
  host-side dispatch + template bool，而不是在 kernel 内读环境变量。
- 构建恢复通过：
  `cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench`。
- Granite 4.0 1B rot-kv 当前 quick sanity（RTX 5050 Laptop，
  `LLAMA_KV_MIXED_VEC_RAW=1 LLAMA_KV_MIXED_VEC_MAIN=1 HP_SINK=64 HP_RECENT=256`）：
  - LP-only pp2048：约 `1559 tok/s`
  - full `oscar_int2` pp2048：约 `1323 tok/s`
  - full `oscar_int2` pp8192：约 `585 tok/s`
- 结论：当前源码可编译、可运行，但 8K 速度仍低于历史好点 `~650-705 tok/s`，
  更远低于 `baseline_bf16` 的 `~3887 tok/s`。目标仍未完成；继续优化应聚焦
  LP bulk/full mixed pipeline 的结构成本，而不是恢复 device-side env 诊断。
- 后续同日复测显示 8K 单次波动较大：default full 可在约 `583-645 tok/s` 间波动，
  LP-only 约 `691 tok/s`，`LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=1` 同批没有稳定收益
  （2K 还可能回退），因此不要把 HP mask skip 改成默认。
- 尝试把 OSCAR2 V half2 fast path 中的 `V_oscar2 = (const block_oscar2_kv *)(V+k*nb21)`
  从维度循环提升到每个 key 一次，构建通过但无收益/负收益：
  pp2048 LP/full 约 `1531/1261 tok/s`，pp8192 LP/full 约 `642/590 tok/s`；
  已回退。不要重复这类指针提升微优化。
- 小幅保留优化：HP combine warp kernel 将 `logit_softcap != 0` 的 runtime 分支改成
  host-side dispatch 的 template bool。构建通过；Granite softcap=50 路径 pp2048 full
  约 `1336 tok/s`，pp8192 full 单跑约 `648 tok/s`。收益不大，但语义等价且避免每个
  HP key 的 runtime 分支；真正瓶颈仍是 LP bulk 约 `~690 tok/s` 的天花板。
- 尝试 `LLAMA_KV_MIXED_VEC_RAW_UNNORM_DST=1`：让 LP 写 raw numerator 到 `dst`，
  HP combine 直接按未归一化 numerator 合并，理论上省 LP 归一化和 combine 反乘。
  实测强负收益：pp2048 default/unnorm 约 `1413/1233 tok/s`，pp8192
  default/unnorm 约 `665/471 tok/s`；已回退。说明瓶颈不是简单的除法/乘法或
  `lp_num` buffer 分配。
- 尝试 host-gated `LLAMA_KV_MIXED_VEC_OSCAR2_KQ_LANES=2/4`，把 OSCAR2 KQ 从
  默认 1 lane 改成多 lane 分摊 D=128。结果不适合作为默认：pp2048 LP-only
  lanes 1/2/4 约 `1422/746/1385 tok/s`；pp8192 LP-only 约 `616/691/461 tok/s`，
  lanes=2 full pp8192 约 `661 tok/s`，没有超过默认好点且 2K 严重回退。已回退；
  继续说明简单改变 KQ lane 粒度不是突破方向。
- 复测临时 raw mixed `ncols=2` dispatch（当前 softcap-template 版本）仍为负收益：
  pp2048 full ncols 4/2 约 `1357/1107 tok/s`，pp8192 full 约 `657/401 tok/s`。
  已回退；保持 raw mixed 默认 ncols=4，不再重复 ncols=1/2/8 方向。
- 审计旧 dedicated LP kernel 后确认不应复活：它使用 float2 VKQ accumulator、
  `nthreads_KQ=2` 和自有 shared combine，正好踩中此前已验证的慢路径；generic vec 的
  half2 V 累加仍是当前更好的 LP 基础。
- 尝试 OSCAR2 KQ half2 pair 累加（把 4 个 half 标量转 float 的 KQ LUT 累加改成
  两个 half2 pair）。LP-only 有正信号：pp2048 `1405 -> 1573 tok/s`，pp8192
  `679 -> 712 tok/s`；full pp2048 `1370 -> 1447 tok/s`，但 full pp8192 重复后
  基本持平（约 `648/647`），默认化后单跑又回到 pp2048 `1332`、pp8192 `588`。
  因收益没有在 full 8K 稳定站住，已回退；不要默认保留，除非后续结合更大的 LP
  主循环重构一起复测。
- 尝试在 raw mixed launcher 内用 CUDA event 做 `LLAMA_KV_MIXED_VEC_RAW_TIMING=1`
  分段计时。p512 能打印每层 LP/HP 时间，稳定层约 `lp_ms ~1.98`、`hp_ms ~2.56`，
  说明短上下文 full 路径里 HP combine/第二段开销不小；但 bench 后半段触发 CUDA
  error，疑似 event synchronize/destroy 与当前 async graph 调度不兼容。计时代码已回退，
  不要把 CUDA event 计时留在默认/bench 路径；若后续需要分段证据，改用独立 microbench
  或外部 profiler。
- 尝试 HP combine half2 pair 版本（env `LLAMA_KV_MIXED_VEC_HP_H2=1`）：p512
  default/h2 约 `1309/1338 tok/s`，pp2048 约 `1314/1439 tok/s`，但 pp8192
  重复后不稳定且不优于默认好点（约 `669/660`，另一轮 `590/580`）。已回退；
  该方向可解释短上下文 HP combine 成本，但不能作为 8K/长上下文默认优化。
- 尝试 HP combine prefill-range 快路径（临时 env
  `LLAMA_KV_MIXED_VEC_HP_PREFILL_RANGE=1`）：用单序列连续 prefill 的 HP sink/recent
  位置规则替代 F32 HP mask 读取。构建通过，2+2 smoke 输出 `2+2 is 4`；p512
  default/range 约 `1293/1369 tok/s`，pp2048 约 `1380/1467 tok/s`，但 pp8192
  多轮仅 `658/661`、`665/668`，最后一轮 range 掉到 `590 tok/s`。该逻辑语义也只适合
  bench 式单序列连续 prefill，不能覆盖通用 mask/seq 场景。已完全回退；不要继续把
  HP mask/prefill range 当作主优化方向。
- 尝试把 OSCAR2 KQ shared centroid LUT 从 half 改成 float（仅 D<=256，避免 D=512
  模板 shared memory 超限）。构建通过后 pp2048 约 `1381 tok/s`，与默认 `1384`
  持平；pp8192 单次 `667`，重复后 `578/664`，没有稳定正收益，且 shared memory 增加
  可能伤 occupancy。已回退；不要继续在 KQ LUT 存储精度上做小修。
- 尝试 LP normalized output 直接写 half 中间 buffer、HP combine 读 half（临时
  `LLAMA_KV_MIXED_VEC_RAW_HALF_LP=1`），用于验证 full raw mixed 掉速是否主要来自
  LP->HP 的 float 中间结果带宽。构建通过，但 pp2048 default/half 约
  `1385/1375 tok/s`，pp8192 约 `668/663 tok/s`，均无收益。已回退；说明当前 full
  path 瓶颈不是简单中间 buffer 带宽，后续应转向真正 fused/tiled LP+HP 主循环或专用
  LP bulk kernel，而不是压缩 LP 输出格式。
- 尝试 OSCAR2 KQ 双 query unpack 复用（临时 `LLAMA_KV_MIXED_VEC_RAW_PAIR_KQ=1`）：
  同一个 K 的 `qs/rs` unpack 同时喂给两个 query 的 centroid LUT，避免此前 4-query
  全展开的寄存器压力。构建通过，pp2048 default/pair 约 `1375/1392 tok/s` 有很小正信号，
  但 pp8192 `671/566 tok/s` 明显负收益。已回退；继续说明 K unpack 复用仍会在长上下文被
  寄存器压力/occupancy 或 instruction mix 抵消，不能作为主优化方向。
- 尝试把 generic OSCAR2 V half2 path 的 `V_rows_per_thread` 从 4 改为 8，并在一次循环中
  decode 8 个 V 元素。构建通过，pp2048 约 `1419 tok/s` 有小正信号，但 pp8192 约
  `631 tok/s`，低于近期默认 `660-670 tok/s`。已回退；说明单纯增大每线程 V decode
  粒度会在长上下文因寄存器/occupancy 损失回退，真正 V tile 需要跨 query/warp 级重排，
  不是每线程多解几个元素。
- 尝试 OSCAR2 generic V 累加用 warp shuffle 直接取 `KQ_reg`，减少
  `KQ[j*nthreads + k]` shared store/load 往返。构建通过，pp2048 约 `1389 tok/s`
  基本持平，pp8192 约 `656 tok/s`，低于近期默认好点。已回退；说明 shared prob
  往返不是当前主要瓶颈，或 shuffle 开销/依赖抵消收益。
- 复测 full vs LP-only（`LLAMA_KV_MIXED_VEC_RAW_LP_ONLY_DIAG=1`）：同一构建下
  pp2048 full/LP-only 约 `1342/1524 tok/s`，pp8192 约 `570/710 tok/s`。HP combine
  仍有 12-20% 可见成本，但不是唯一瓶颈。检查 HP tensor 形态后确认当前
  `sink=64 + recent=256` 时 `k_hp->ne[1]=320` 且真实 active HP slots 也是 320；
  `get_k_hp/get_v_hp/get_hp_kq_mask` 的 256 padding 没有在该配置下额外多扫。因此不能靠
  去 padding 减少 HP loop；继续缩 HP/mask range 会回到此前不稳定且语义受限的方向。
- 尝试 K-side 4-level upper-bound 诊断：在 generic raw LP 的 OSCAR2 KQ 路径中临时忽略
  `rs` high bit，只用 `qs` 低 2bit 查 4 个 centroid，以测试 K residual 解码/8-level
  查表是否是主要速度瓶颈。构建通过，2+2 smoke 正常输出 `2 + 2 equals 4`，但
  pp2048/pp8192 仅约 `1453/684 tok/s`，没有稳定超过正确 8-centroid 路径的好点，
  且该诊断有明显 K 质量风险。已回退；不要把“去掉 K residual/降成 4-level K”作为
  当前 oscar_int2 默认或主要优化方向。
- 尝试 env-gated fast exp 诊断（临时 `LLAMA_KV_MIXED_VEC_FAST_EXP=1`）：只覆盖 raw
  mixed 实际命中的 `logit_softcap && normalize_lp` LP kernel 与 HP combine，将主要
  softmax `expf` 替换为 `__expf`。构建通过，但 pp2048 default/fast-exp 约
  `1475/1373 tok/s`，明显负收益；未继续跑 8K，代码已回退。说明当前瓶颈不是可通过
  fast math 近似指数函数解决，且不应引入额外质量变量。
- 尝试 env-gated softcap `tanhf` 近似诊断（临时 `LLAMA_KV_MIXED_VEC_APPROX_SOFTCAP=1`）：
  只覆盖 raw mixed 的 Granite `logit_softcap && normalize_lp` 路径，用
  `x/(1+abs(x))` 近似 `tanhf(x)` 以测试 softcap 数学函数是否是大瓶颈。构建通过，但
  pp2048 default 同批异常低约 `665 tok/s`，approx 约 `1399 tok/s` 也没有超过此前正确
  路径好点（约 `1475 tok/s`），且近似 softcap 会直接引入质量变量。已回退；不要把
  softcap/tanh 近似作为 oscar_int2 主优化方向。
- 尝试 HP combine 单独 8-query tile 诊断（`LLAMA_KV_MIXED_VEC_HP_NCOLS8=1`）：
  LP raw kernel 仍保持默认 `ncols=4`，只把 HP correction kernel 从 `ncols=4` 改成
  `ncols=8`，用于测试 full path 掉速是否主要来自 HP combine 的 query tile 粒度。
  构建通过；pp2048 default/HP_NCOLS8 约 `1459/1530 tok/s` 有小正信号，但 pp8192
  复测 default/HP_NCOLS8 约 `652/653 tok/s`，基本持平，未突破 700 gate。该开关可保留
  作为诊断，但不要默认化；继续说明 HP combine 局部 tile 粒度不是主要突破点。
- 尝试 OSCAR2 V pair-query accumulation 诊断（临时
  `LLAMA_KV_MIXED_VEC_OSCAR2_V_PAIR_ACCUM=1`）：只在 generic raw LP 的 OSCAR2 half2 V
  累加内层，把 `ncols=4` 的 query 循环手写展开，试图减少索引/循环开销；KQ/softmax/HP
  均不变。构建通过，LP-only pp2048 default/pair 约 `1495/1496 tok/s` 持平，LP-only
  pp8192 单次 `621/705 tok/s` 有好点；但 full pp8192 pair 约 `585 tok/s`，低于同批
  default full 约 `679 tok/s`。已回退；不要把 V query-loop 手写展开作为默认，LP-only
  单次好点不能代表 full mixed 收益。
- 尝试 OSCAR2 D=128 LP tail shared-reduce unroll 诊断（临时
  `LLAMA_KV_MIXED_VEC_OSCAR2_TAIL_UNROLL=1`）：只在 generic raw LP 尾部把
  `nwarps * V_cols_per_iter` 的 shared `KQ` 汇总循环手写展开成 8 项累加，KQ/softmax/V
  解码均不变。构建通过，但 LP-only pp2048 default/tail-unroll 约 `1644/1461 tok/s`，
  明显负收益；已回退，未继续跑 8K。说明尾部小循环不是瓶颈，手写展开增加的指令/寄存器
  压力超过收益。
- 尝试 LP bulk 长度上限诊断（临时 `LLAMA_KV_MIXED_VEC_LP_TAIL_LIMIT=N`）：在 raw mixed
  launcher 中限制 LP raw kernel 的 `ne11_lp_eff`，HP correction 仍照常执行，用于量化
  LP bulk 扫描长度对 8K full prefill 的影响。该实现截断的是 LP 前缀长度，不是正确的
  recent/tail 语义，因此只作速度上限，不能默认或用于质量结论。构建通过；pp8192 full
  default/limit2048/limit1024/limit512 约 `680/1218/1701/1982 tok/s`。已回退。结论：
  LP bulk 扫描长度是当前最大杠杆，继续优化常数难以接近 BF16；后续若要真正提速且保质量，
  应转向正确语义的分层/稀疏 LP 访问、two-tier/residual 格式，或更深的 LP+HP fused
  tiled 主循环，而不是继续调 helper/小循环。
- 尝试 dedicated LP raw kernel 上的全上下文 stride 采样诊断（临时
  `LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP=1 LLAMA_KV_MIXED_VEC_LP_STRIDE=N`）：只在旧
  dedicated LP 诊断 kernel 中把 `key=(k0+i)*stride`，覆盖整个 LP 老上下文但减少采样 key
  数；HP correction 仍照常执行。构建通过，2+2 smoke 在 stride=4 下输出正常
  `2+2 is 4`，无重复退化。pp8192 full dedicated stride 1/2/4 约 `417/745/996 tok/s`。
  已回退。结论：稀疏覆盖比前缀截断更接近可用方向且有速度空间，但旧 dedicated kernel
  本身太慢，不能复活为默认；若继续，应在 generic/tiled 快路径中实现正确的 sparse/two-tier
  LP 访问，并做数据集质量验证。
- 尝试 generic raw LP 快路径上的全上下文 stride 采样诊断（临时
  `LLAMA_KV_MIXED_VEC_LP_STRIDE=N`）：通过 CUDA constant 只让 raw
  `OSCAR2_KV/OSCAR2_KV` LP kernel 按 `key_physical=key_logical*stride` 读取 K/V/mask，
  HP correction 保持不变，用于验证 sparse LP 覆盖在当前最快 generic 路径中的速度上限。
  单进程 pp8192 full default/stride2/stride4 约 `669/1096/1512 tok/s`；stride4 的
  2+2 smoke 输出 `2 + 2 = 4.`，无 `2. 2. 2...` 退化。该路径语义是采样 attention，
  不是完整 BF16 HP + INT2 bulk，不能作为 oscar_int2 默认或质量结论；代码已回退，构建通过，
  回退后默认 pp8192 full 约 `648 tok/s`。结论：LP bulk 覆盖成本是当前最大瓶颈，下一步
  应设计有质量约束的 sparse/two-tier/residual LP bulk 或真正 fused tiled 主循环，而不是
  保留简单 stride 采样开关。
- 新增 env-gated two-tier LP 访问候选（非默认）：
  `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_STRIDE=N` +
  `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_TAIL=M`。只作用于 raw normalized
  `OSCAR2_KV/OSCAR2_KV` LP kernel：老 LP 前缀按 stride 采样，近端 LP tail 全量保留，
  BF16 HP correction 仍按原路径合并。该实现通过 CUDA constant 在 kernel 内把 logical key
  映射到原始 K/V/mask physical key，不改变外部 tensor/HP combine 形态；默认 stride=1
  保持完整 LP 语义。
  - 构建通过；`stride=4, tail=1024` 的 2+2 smoke 输出 `4`，无重复退化。
  - 同场 pp2048 default/two-tier 约 `1246/1327 tok/s`。
  - pp8192 default 单次约 `474 tok/s`（低波动点），`stride=4, tail=1024` 单次约
    `872 tok/s`；三次重复约 `878/855/904 tok/s`，稳定超过 8K `700 tok/s` gate。
  - 更保守参数 `stride=2, tail=1024` pp8192 约 `608 tok/s`；
    `stride=4, tail=2048` pp8192 约 `625 tok/s`，未过 gate。
  - 小质量 smoke（CLI eval，ctx 4096，20 题）：GSM8K baseline_bf16/two-tier
    `7/20` vs `6/20`；GPQA baseline_bf16/two-tier `4/20` vs `3/20`。这是小样本，
    只说明未直接崩，不能作为最终质量结论。
  - 当前状态：two-tier 是第一个同时过 8K gate 且小样本质量未明显崩的速度候选，但它仍是
    sparse LP 近似，未证明完整数据集质量，也远低于 BF16 8K 约 3.9k tok/s；不要默认化。
    下一步应扩大质量验证，并把固定 stride 改成更有质量依据的 two-tier/residual/importance
    选择，或继续做真正 fused tiled LP 主循环来追 BF16。
- 继续 two-tier 诊断：
  - `stride=8, tail=1024` pp8192 约 `1021 tok/s`，比 `stride=4, tail=1024`
    更快但仍远低于 BF16；`stride=8, tail=2048` 约 `653 tok/s`，保留太多 LP tail 后
    不过 700 gate。
  - 新增 `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_MODE=end|mid`，控制前缀每组采样点；
    默认仍是 group-start。`mode=end, stride=4, tail=1024` pp8192 约 `884 tok/s`，
    与 start 模式基本相同；GSM8K 20 题仍为 `6/20`，没有相对 start 模式的质量改善。
  - 结论：简单改变采样点不能明显改善质量；若继续 sparse/two-tier，应转向
    importance/residual 选择或保存额外低成本 summary，而不是只调 stride offset。若追速度超过
    BF16，仅靠减少 key 数仍不够，需要进一步降低 per-sampled-key 成本或做更深的 fused/tiled
    主循环。
- 新增 `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_WEIGHTED=1` 诊断：让 sampled LP prefix 的每个
  key 按其代表的 group size 放大 softmax 概率贡献（同时影响 denominator 与 V numerator），
  tail 仍全量、不加权。`stride=4, tail=1024, mode=end, weighted=1` pp8192 约
  `886 tok/s`，速度与 unweighted 基本相同；但 GSM8K 20 题仍 `6/20`，GPQA 20 题仍
  `3/20`，没有相对 unweighted 的小样本质量收益。结论：简单 group-size weighting 不能修复
  sparse LP 质量差距，继续应转向 importance/residual 选择或更强的 LP summary，而不是
  继续调固定权重。
- two-tier sparse 速度上限诊断：更激进 `mode=end` 下，pp8192
  `stride=16, tail=512` 约 `1091 tok/s`，`stride=16, tail=1024` 约 `1142 tok/s`，
  `stride=32, tail=512` 约 `1566 tok/s`，`stride=32, tail=1024` 约 `1183 tok/s`。
  同参数 `stride=32, tail=512` 下 full/LP-only 约 `1568/1864 tok/s`，HP combine/合并约
  15-20% 成本，但 LP-only 本身也远低于 BF16 8K 约 3.9k。结论：固定 sparse 在当前
  generic raw mixed kernel 中的速度天花板约 1.5-1.9k，已经不能靠继续加大 stride 达成
  “速度超过 BF16”。下一步必须转向专用 fused/tiled LP 主循环，减少 per-sampled-key 固定成本，
  并把 HP correction 合进同一个 tile/online-softmax 流程；sparse/importance 只能作为质量/显存
  策略，不能单独解决速度目标。
- 在 sparse/two-tier 条件下复测旧 tile 粒度开关：`stride=32, tail=512` 下
  `LLAMA_KV_MIXED_VEC_NCOLS=8` pp8192 约 `1292 tok/s`，`LLAMA_KV_MIXED_VEC_HP_NCOLS8=1`
  约 `1559 tok/s`，二者同时开约 `1291 tok/s`；仍无突破，ncols=8 继续负收益，
  HP_NCOLS8 基本持平。修正 `LLAMA_KV_MIXED_VEC_RAW_LP_ONLY_DIAG` 判断：现在只有 env
  非空且非 `0` 才触发 LP-only，避免 `=0` 误触发诊断路径。
- 下一步专用 fused/tiled kernel 边界：不要复活旧 `flash_attn_ext_mixed_oscar2_lp_raw_vec`
  作为基础；它历史上完整 D=128 correctness 修复后明显慢于 generic。新的最小实现应只覆盖当前
  真实目标组合：D=128、`OSCAR2_KV/OSCAR2_KV` LP、BF16 HP、`parallel_blocks=1`、
  `ncols=4`、Granite `logit_softcap`、two-tier key mapping。先把 LP raw normalized 输出、
  `lp_meta` 写回和独立 HP combine kernel 融成一个 kernel；若仍低于 LP-only 上限，再继续把
  KQ/V 做成更深的 tiled/shared 主循环，减少 per-sampled-key 固定成本。
- 尝试复用旧 `flash_attn_ext_mixed_oscar2_f16_vec` 做 raw fused two-tier prototype：补了
  Granite softcap、F16 LP mask、two-tier key mapping，并通过临时
  `LLAMA_KV_MIXED_VEC_RAW_FUSED_TWO_TIER=1` 触发。构建通过，但 `stride=32, tail=512`
  下 pp2048 约 `877 tok/s`、pp8192 约 `751 tok/s`，明显慢于 generic raw two-tier
  full 约 `1568 tok/s`。已撤回入口和旧 fused kernel 参数改动。结论：旧 fused 结构不能作为
  新 fused/tiled 基础；下一步必须新写专用 D=128 raw two-tier kernel，直接围绕当前 generic
  快路径的 K LUT / half2 V 累加组织重排，而不是复用旧 fused mixed kernel。
- 保留优化：generic raw OSCAR2 KQ subgroup 从 1 lane 改为 2 lane。此前完整路径好点曾偏向
  1 lane，但在 two-tier sparse 下每个 sampled key 的固定成本更重要，2 lane 有明显正收益：
  `stride=32, tail=512, mode=end` pp8192 full/LP-only 从约 `1568/1864 tok/s` 提升到约
  `1783/2401 tok/s`；较保守 `stride=4, tail=1024, mode=end` pp8192 约 `1074 tok/s`
  （此前约 `855-904`），pp2048 约 `1360 tok/s`。默认完整 no-two-tier pp8192 约 `595 tok/s`
  没有明显收益且仍低于历史好点，但当前候选方向是 two-tier sparse，因此暂时保留。目标仍未完成：
  即使 LP-only 约 2.4k，也低于 BF16 8K 约 3.9k，后续仍需专用 tiled/fused kernel。
- 继续测试 OSCAR2 KQ subgroup=4（仅改 generic raw `nthreads_KQ`，构建通过）：
  `stride=32, tail=512, mode=end` pp8192 full/LP-only 约 `1516/1774 tok/s`，
  明显慢于 subgroup=2 的约 `1783/2401 tok/s`，已回退到 2 lane。回退后同参数 full
  sanity 约 `1947 tok/s`，说明状态干净。结论：two-tier 下继续增加 KQ lane 会带来寄存器/
  occupancy 负收益，不要重复 subgroup=4/更大 lane 方向；下一步应减少 per-sampled-key
  整体循环/写回/HP combine 成本，或新写 D=128 专用 tiled/fused two-tier 主循环。
- 极端 two-tier 稀疏速度上限探针（不作为质量候选）：顺序单跑 `stride=64, tail=256,
  mode=end` pp8192 约 `2183 tok/s`，`stride=128, tail=128, mode=end` 约
  `2449 tok/s`。二者仍远低于 baseline BF16 8K 约 `3887 tok/s`。另外一次并行误跑
  得到的 LP-only/极端 sparse 数字因 GPU 互抢不可信，已丢弃。结论：即使把 LP bulk
  采样到很激进，generic raw mixed 的 softmax/V/reduce/writeback/HP combine 固定成本也
  追不上 BF16；继续单纯调 stride/tail 不能完成目标。必须转向专用 D=128 kernel，减少
  每个 sampled key 和每个 query tile 的固定开销，并把 HP correction 合并得更轻。
- 试过 generic raw LP 尾部 fast-combine 专用化（临时
  `LLAMA_KV_MIXED_VEC_FAST_COMBINE=1`，只针对 D=128 OSCAR2 raw normalized、
  `gridDim.y=1`，把通用 `KQ_max_shared[ncols][32]` 归约改成 4-warp 专用归约）。
  构建通过，但 `stride=128, tail=128, mode=end` pp8192 LP-only 从默认约
  `2852 tok/s` 掉到约 `2459 tok/s`，负收益；代码已全部撤回，`rg FAST_COMBINE`
  无命中，构建再次通过。结论：尾部 combine 小修不是主要突破口，且容易破坏现有编译器/
  shared-memory 布局；后续不要继续在 generic vec 尾部归约上做微调，应直接写新的
  D=128 OSCAR2 tiled/fused 主循环。
- 旧 `flash_attn_ext_mixed_oscar2_lp_raw_vec` dedicated LP 路线再次排除：审计发现该入口
  在 two-tier 下没有按 logical key 映射到 physical K/V/mask key，之前任何 two-tier 速度
  数字都不是同一语义。临时补齐 physical mapping 后构建通过，但
  `LLAMA_KV_MIXED_VEC_RAW_DEDICATED_LP=1` + `stride=128, tail=128, mode=end`
  pp8192 LP-only 超过 60 秒仍无结果，被中断，明显远慢于 generic raw LP-only 约
  `2.8k tok/s`。补丁已撤回，构建再次通过。结论：不要再修旧 dedicated LP；v2 必须是
  新 kernel/新主循环，而不是基于这个旧结构做局部修补。
- 试过 two-tier tile 内 physical key 预计算（临时 `LLAMA_KV_MIXED_VEC_KEY_PREMAP=1`）：
  在每个 128-key tile 开头把 logical→physical 映射写到 shared，KQ/V 复用。构建通过，
  但 `stride=128, tail=128, mode=end` pp8192 LP-only 从默认约 `2889 tok/s` 掉到约
  `1766 tok/s`，大幅负收益；代码已撤回，`rg KEY_PREMAP` 无命中，构建再次通过。
  结论：重复 key 映射不是主瓶颈，额外 shared 写读和同步会严重破坏当前 fast path；后续
  不应继续做这类整数辅助逻辑微调。
- 试过 OSCAR2 generic raw V `V_rows_per_thread=8`（每 lane 一次处理 8 个 V 元素，
  增加对应 inline 8-value decode），构建通过。`stride=128, tail=128, mode=end`
  pp8192 LP-only 约 `2873 tok/s`，回退到默认 rows=4 后同组约 `2898 tok/s`；
  full 单次 rows=8 约 `2582 tok/s`、rows=4 约 `2339 tok/s`，但 LP-only 没有收益且
  full 受 HP combine/波动影响，不足以证明正收益。rows=8 代码已撤回并构建通过。结论：
  rows=4 仍是更稳的 V decode/accumulate 粒度；减少 V loop 迭代数会被寄存器/布局成本抵消。
- 试过 KQ probability warp-shuffle 读取（临时 `LLAMA_KV_MIXED_VEC_KQ_SHFL=1`）：
  V 侧不从 shared `KQ[j*nthreads+k]` 读概率，而用 `__shfl_sync` 从对应 KQ producer lane
  的 `KQ_reg[j]` 取值；为降低风险仍保留 shared 写。构建通过，`stride=128, tail=128,
  mode=end` pp8192 LP-only A/B 波动较大：默认约 `2682/2663 tok/s`，KQ_SHFL 约
  `2912/2636 tok/s`，没有稳定正收益；代码已撤回，`rg KQ_SHFL` 无命中，构建再次通过。
  结论：单独把 shared 读换成 shuffle 不足以突破，真正要改必须连同 shared 写、V 累加布局
  和在线 softmax 状态一起重排，而不是保留原结构只换读取方式。
- 新增 OSCAR2 v2 独立落点（默认关闭）：`fattn-oscar2-v2.cu/.cuh`，并在
  `ggml-cuda/CMakeLists.txt` 显式编译 `fattn-oscar2-v2.cu`。入口由
  `LLAMA_KV_MIXED_VEC_OSCAR2_V2=1` 触发，目前只是 host stub，返回 `false` 后回退
  generic raw LP，不改变默认行为。构建通过，debug bench 确认能拿到真实参数：
  `q=512 k_eff=129 ncols=4 norm_lp=1 softcap=50 stride=128 tail=128 original=256
  effective=129 mode=127`。下一步应在该新文件内加入 D=128/ncols=4/normalized/two-tier
  kernel skeleton，再逐步实现真正 tiled/fused 主循环；不要继续把 v2 实验塞进
  `fattn-vec.cuh` 主路径。
- v2 skeleton 继续推进：`fattn-oscar2-v2.cu` 中新增 no-op CUDA kernel，
  仅在 `LLAMA_KV_MIXED_VEC_OSCAR2_V2_FORCE=1` 下返回 true 并写 `dst/meta`，普通
  `LLAMA_KV_MIXED_VEC_OSCAR2_V2=1` 仍返回 false 回退 generic。构建通过；p512
  LP-only FORCE sanity 能跑完（输出无正确性意义，只验证 launch/dst/meta/return-true
  路径）。下一步把 no-op 替换成最小实际 LP kernel：先只支持 D=128、ncols=4、
  normalized LP、two-tier，目标是 correctness/smoke 后再谈速度。
- v2 minimal LP kernel 已替换 no-op：`LLAMA_KV_MIXED_VEC_OSCAR2_V2_FORCE=1` 下
  启动 row-per-block scalar kernel，覆盖 D=128、OSCAR2 K/V scalar dequant、F16 mask、
  Granite `logit_softcap=50`、two-tier physical key mapping，并写 normalized LP output
  和 `lp_meta`。构建通过；p512 LP-only FORCE sanity 能跑完，速度约 `1143 tok/s`。
  该版本用于语义/参数/launch 打通，结构上每 row 一个 block、每 key 用 atomic 汇总 KQ，
  明显不可能是最终速度路径；不要拿它做性能结论。下一步需要把该 minimal kernel 改成
  ncols=4/tile kernel：block 处理 query tile，KQ 和 V 累加共用 tile 内 online softmax，
  移除 per-key atomic 和 row-per-block 结构。
- v2 FORCE tile4 继续推进（2026-06-18）：`fattn-oscar2-v2.cu` 的 FORCE kernel 改为
  4-query tile block，并保留默认关闭；普通 `LLAMA_KV_MIXED_VEC_OSCAR2_V2=1` 仍会回退
  generic raw，不改变默认 `oscar_int2`。构建通过。第一版 tile4 每个 query 对 4 个 warp
  partial 做 atomic 汇总：p512 LP-only 约 `1549 tok/s`，p2048 单次约 `1679 tok/s`，
  pp8192 LP-only 约 `2414 tok/s`，full pp8192 约 `2253 tok/s`。尝试把 atomic 改为 shared
  warp-sum reduction 后 p2048 掉到约 `810 tok/s`，已撤回；不要重复这个 shared reduction
  小改。随后改成 warp-per-query KQ，并修正为每个 query warp 覆盖完整 D=128
  (`lane, lane+32, lane+64, lane+96`)：p2048 LP-only 约 `2091 tok/s`，pp8192 LP-only
  约 `2711 tok/s`，full pp8192 约 `2194 tok/s`。该结构在 2K 有明显收益，但 8K 仍略低于
  generic LP-only 约 `2800 tok/s`，full 仍远低于 BF16 约 `3887 tok/s`，不能默认。短
  `2+2` smoke 无 CUDA 报错、未复现 `2. 2. 2...`，但回答错误（偏向 `1/3`），说明
  `FORCE + stride=128/tail=128` 仍只是速度/结构实验，不是质量可用路径。下一步若继续 v2，
  应做真正 V tile/prob×V staging 或更保守 two-tier 质量参数下的专用 kernel，而不是把
  当前 FORCE 路径接成默认。
- v2 FORCE 后续 V/K 小改（2026-06-18）均未形成可保留默认方向：
  - 尝试在 v2 FORCE 中加入 `LLAMA_KV_MIXED_VEC_OSCAR2_V2_V_H2=1` half2 V numerator：
    p2048 LP-only 基本持平（约 `1976 -> 1982 tok/s`），pp8192 LP-only 负收益
    （约 `2679 -> 2611 tok/s`），已撤回，避免新增实验噪音。
  - 尝试每个线程负责 4 个 V 元素、用 `oscar2_dequantize_4_v_f2` + `float4`
    numerator/output：p2048 LP-only 约 `1802 tok/s`，慢于 v2 scalar V 约 `2.0k+`，
    已撤回。减少 V decode 调用数会被寄存器/活动线程减少抵消。
  - 尝试 v2 K-side shared `Q*centroid` LUT：p2048 LP-only 约 `1925 tok/s`，
    慢于无 LUT；shared 表访问/占用成本高于省下的标量 centroid 计算，已撤回。
  - 在更可能保持质量的 two-tier 参数 `stride=4, tail=1024, mode=end, weighted=1`
    下，generic LP-only p2048 约 `1621 tok/s`，v2 FORCE p2048 约 `1089 tok/s`；
    当前 v2 tile4 不适合作为质量参数路径继续小修。后续应回到 generic raw fast path
    和 full HP combine 掉速，或重写真正 tile kernel，而不是继续在当前 v2 FORCE 上替换
    accumulator 类型/小 LUT。
- generic raw weighted two-tier 小优化（2026-06-18）：把 `weighted=1` 的 prefix group
  size 从每个 sampled key 做 `min/max/group_start` 改成预先计算
  `prefix_full_groups` 和 `prefix_last_group_size`，大多数 key 直接乘常数 `stride`，语义等价。
  构建通过。`stride=4, tail=1024, mode=end, weighted=1` 下 p2048 LP-only 约
  `1632 tok/s`，pp8192 LP-only 约 `1324 tok/s`，相比同批修改前 p2048 约 `1621`、
  pp8192 约 `1276` 小幅正收益；但 full pp8192 仍约 `1132 tok/s`，与修改前约
  `1136 tok/s` 基本持平。结论：LP bulk 控制开销略有下降，但 full 路径瓶颈没有随之
  改善；下一步应看 HP combine/full pipeline 或重新设计更完整 tile kernel。
- generic raw full/HP 诊断（2026-06-18）：`stride=128, tail=128, mode=end`
  full pp8192 约 `2373 tok/s`；`stride=4, tail=1024, mode=end, weighted=1`
  full pp8192 约 `1132-1136 tok/s`，LP-only 约 `1324 tok/s`。把
  `HP_RECENT=0`（仍保留 `HP_SINK=64`）后 full pp8192 约 `1239 tok/s`，说明 HP
  recent 扫描吃掉一部分，但 64 sink combine 和二次读写仍有固定成本。已有
  `LLAMA_KV_MIXED_VEC_HP_NCOLS8=1` 在质量参数下无收益：recent=256 约 `1122 tok/s`，
  recent=0 约 `879 tok/s`，不要把 HP_NCOLS8 当优化方向。下一步继续优先优化
  `stride=4/tail=1024/weighted` 的 LP bulk，而不是 HP_NCOLS8 或 v2 小修。
- generic raw two-tier physical-key helper（2026-06-18）：将 KQ/V 两处重复
  logical→physical 映射合并为 `oscar2_two_tier_physical_key()`，并为常见
  `stride=4, mode=end` 使用 shift/add 快路径；语义等价。构建通过。
  `stride=4, tail=1024, mode=end, weighted=1` 下 pp8192 LP-only 约
  `1331 tok/s`，full pp8192 约 `1135 tok/s`，相比上一轮 LP-only 约 `1324`
  只有小幅变化，full 基本不动。结论：整数映射不是主瓶颈；质量参数要继续提速，
  需要减少实际 LP bulk 工作量或重写更完整的 tiled kernel，而不是继续微调 key 映射。
- quality 参数下 ncols 探测（2026-06-19）：`stride=4, tail=1024, mode=end,
  weighted=1`，pp8192 上默认 `ncols=4` 仍最好。同一构建下 `ncols=2`：
  LP-only 约 `1334 tok/s`、full 约 `1104 tok/s`；`ncols=8`：
  LP-only 约 `805 tok/s`、full 约 `641 tok/s`。不要把 `LLAMA_KV_MIXED_VEC_NCOLS=2/8`
  作为质量参数默认优化方向；继续保持 ncols=4。后续若要提升质量可用速度，应减少
  LP effective token 数并用质量补偿，而不是继续调 query tile 宽度。
- two-tier 参数网格（2026-06-19）：pp8192 full 上测试
  `stride=8/tail=1024`、`8/2048`、`16/2048`、`16/4096`，均 `mode=end,
  weighted=1`。速度分别约 `1311`、`1069`、`1098`、`781 tok/s`；只有
  `stride=8/tail=1024` 比 `stride=4/tail=1024` 的约 `1135 tok/s` 有明显提升。
  但质量小集显示该候选不可用：`runs/oscar_int2_stride8_quality_20260618T161448Z`
  中 baseline BF16 GPQA/GSM8K 为 `3/10`、`4/10`，oscar_int2 stride8 为
  `0/10`、`0/10`；2+2 smoke 能答 4，但数据集输出格式/答案明显崩。结论：
  单纯提高 stride 减少 LP token 数会破坏质量，不能作为最终 `oscar_int2` 方向。
  下一步需要在 `stride=4` 质量参数附近做格式/残差补偿，或实现真正更快的 LP tile
  kernel，而不是继续加大 sparse stride。
- 三段 LP sampling 实验（2026-06-19）：新增 env-gated 参数
  `LLAMA_KV_MIXED_VEC_LP_THREE_TIER_FAR_STRIDE`,
  `LLAMA_KV_MIXED_VEC_LP_THREE_TIER_MID_TOKENS`,
  `LLAMA_KV_MIXED_VEC_LP_THREE_TIER_FAR_MODE`。语义为 far prefix 使用 far stride，
  中近段使用原 `LP_TWO_TIER_STRIDE`，tail 精确；默认不设置时保持原 two-tier 行为。
  构建通过，p512 debug 显示 `effective=160`、`far_sampled=0`、`mid_sampled=32`
  符合短上下文预期。pp8192 上：
  - `far_stride=8, mid_tokens=2048, stride=4, tail=1024, weighted=1`：
    LP-only 约 `1321 tok/s`，full 约 `1179 tok/s`，仅比 two-tier quality 基线
    full 约 `1135` 小幅提升。
  - `far_stride=16, mid_tokens=2048, stride=4, tail=1024, weighted=1`：
    LP-only 约 `1443 tok/s`，但 full 约 `976 tok/s`，负收益。
  结论：远段更稀疏 + 中近段保守不能单独接近 BF16，也没有形成质量/速度候选。
  该 env 可作为诊断保留，但不要默认启用；后续应转向 residual/format 质量补偿或真正
  LP tile kernel。
- stride=8 采样位置/权重质量复测（2026-06-19）：`stride=8, tail=1024,
  mode=mid, weighted=1` 的 2+2 smoke 能正常答 `2+2 is 4`，pp8192 full 约
  `1301 tok/s`，接近 `mode=end` 的约 `1311 tok/s`；但质量小集仍崩：
  `runs/oscar_int2_stride8_mid_quality_20260618T162928Z` 中 baseline BF16
  GPQA/GSM8K 为 `3/10`、`4/10`，oscar_int2 为 `0/10`、`0/10`。不加
  `weighted` 的 `mode=mid` 连 2+2 都输出异常。结论：简单改变 sampled token
  位置或权重不能恢复 stride=8 质量；继续走 sparse 参数调优没有价值，必须做
  residual/format 质量补偿或 kernel 级提速。
- OSCAR INT2 K/V 质量瓶颈复查（2026-06-19）：新增 env-gated 诊断
  `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_V_AVG=1`、`...K_AVG=1`、
  `...WEIGHT_EXP=<alpha>`，只在 raw OSCAR2 two-tier 路径生效，默认不启用。
  结论均不能作为默认方向：
  - `stride=8, tail=1024, mode=mid, weighted=1, V_AVG=1`：
    2+2 smoke 正常，但 p2048 从约 `1447 tok/s` 掉到约 `680 tok/s`；
    GPQA/GSM8K 3+3 小样本中 baseline BF16 为 `1/3`、`1/3`，
    oscar2 mixed 为 `1/3`、`0/3`。运行时 group V 平均太慢，也没有救 GSM8K。
  - `WEIGHT_EXP=0/0.5/1`：p2048 分别约 `1447/1387/1450 tok/s`，
    pp8192 分别约 `1274/959/1275 tok/s`；`alpha=0` 的 3+3 质量仍是
    `1/3`、`0/3`。说明质量崩不是单纯因为 `group_size` mass correction 过强。
  - `K_AVG=1`：2+2 能答 4，但 p2048/pp8192 约 `1108/445 tok/s`，
    3+3 质量仍是 `1/3`、`0/3`。运行时 group K 平均既慢，也没有恢复 GSM8K。
  - 更关键的 no-two-tier 复测：只开 `LLAMA_KV_MIXED_VEC_RAW=1`
    和 `MAIN=1`，不加 sparse/two-tier，oscar2 mixed 3+3 仍是
    GPQA/GSM8K `1/3`、`0/3`。因此当前问题不能继续归因于 two-tier
    sparse 采样；OSCAR2 K/V 本体或 mixed K-side attention score 仍未达质量门槛。
- K/V 分离诊断（2026-06-19）：同一 3+3 小集下，
  `K BF16 + V q2_0`（内部诊断 `oscar_kbf16_vq2`）得到 GPQA/GSM8K
  `1/3`、`1/3`，接近 BF16 小样本；`K q2_0 + V BF16`
  得到 `0/3`、`0/3`；`K q4_0 + V q2_0` 得到 `1/3`、`2/3`。
  结论非常明确：质量主瓶颈是 K-side attention score，不是 V-side。
  下一步应实现 K-side residual/two-tier summary 或 K 更强内部格式、V 保持低位；
  不要继续投入 V helper、V group average、HP window、mask、ncols、CUDA graph
  或 sparse stride/weight 小调。`K q4_0 + V q2_0` 的 8K bench 走慢路径，
  超过约 100s 未出结果后中断；它只作为质量上界诊断，不是速度方案。
- K-side 提精度质量/速度上界（2026-06-19）：新增 D=128 诊断性 vec FA
  template/dispatch `GGML_TYPE_Q4_0 / GGML_TYPE_OSCAR2_KV`
  （`fattn-vec-instance-q4_0-oscar2_kv.cu`，只用于内部 A/B，不作为用户可见
  `oscar_int2` 命名）。构建通过。Granite rot-kv 上：
  - p2048 约 `753 tok/s`，pp8192 约 `256 tok/s`，说明通用 q4_0 KQ +
    oscar2 V 的 vec 路径很慢，不能作为最终速度方案。
  - 3+3 小质量：baseline BF16 GPQA/GSM8K `1/3`、`1/3`；
    `K q4_0 + V oscar2` 为 `1/3`、`2/3`，与 `K q4_0 + V q2_0`
    一样能恢复 GSM8K 小样本。结论：K-side 提精度/残差确实能救质量，
    但必须实现 OSCAR 专用 K residual/K4 fast KQ，而不是直接使用通用 q4_0
    cache K 路径。下一步应设计 `OSCAR2 K 3-bit + residual` 或 K-only 4bit
    内部格式，并直接配 CUDA FA KQ LUT/vec 主循环；V 继续保持低位。
- q4K + oscar2V 慢点定位（2026-06-19）：同一构建 p2048 对照：
  `q4_0/q4_0` 约 `2910 tok/s`，`oscar2/oscar2` 约 `802 tok/s`，
  `q4_0/oscar2` 约 `805 tok/s`，`bf16/oscar2` 约 `901 tok/s`。
  结论：`q4_0/oscar2` 的慢点主要被 oscar2 V decode/accum 或混合 V 布局限制，
  不是 q4 KQ 本身。尝试只对 `K=q4_0,V=oscar2,D=128` 把 oscar2 V lane 从
  16 改成 32：p2048 约 `756 tok/s`、pp8192 约 `234 tok/s`，低于改前
  p2048 约 `804`、pp8192 约 `256`，已回退。下一步不要再调 V lane 粒度；
  若继续 K-side 提精度方案，需要同时给 V 做更快的低位布局/packed accumulation，
  或把最终内部格式改成能复用 q4/q4 高速 V 路径的 K-residual 组合。
- q4/q4 上界与 oscar2 V decode 诊断（2026-06-19）：`q4_0/q4_0`
  在 rot-kv Granite 上 pp8192 约 `3978 tok/s`，超过 BF16 历史约 `3887 tok/s`；
  3+3 小质量为 GPQA/GSM8K `1/3`、`1/3`，与该小样本 BF16 持平。
  每 128 维 KV 估算：BF16 K/V pair `512B`，q4/q4 `144B`，q4K+oscar2V
  `124B`，oscar2/oscar2 `104B`；因此 K 提到 q4-style 仍有充足显存优势。
  尝试过只在 `q4_0/oscar2` 诊断中忽略 oscar2 V 的 high `rs` bit（low2 V decode）
  来衡量 3-bit decode 成本：显式 off/on p2048 均约 `807-809 tok/s`，无稳定速度收益；
  诊断代码已回退。结论：q4/oscar2 慢不是单个 high-bit decode 成本，而是 oscar2 V
  的整体布局/accum 组织不如 q4/q4 高速路径。下一步应实现 q4-style packed V
  accumulation 或重新设计内部 KV 格式，使 K-side 精度提高的同时复用高速 V 组织。
- Staged FA 重大进展（2026-06-19）：确认 q4/q4 之所以快，是因为 selector 没有强制
  vec，而是可以走 `launch_fattn` 的 f16 staging + tile/MMA 路径；而 q2/oscar2 曾被
  hard-code 到 vec。新增 env gate `LLAMA_KV_OSCAR2_ALLOW_STAGED_FA=1`，允许
  oscar2 继续走 staged FA 决策；同时补了 OSCAR2_KV 的 K/V role-aware contiguous
  to-f16 converter（K 用 `m=0 + K centroids`，V 用 `m + V centroids`），避免通用
  converter 不知道 K/V 角色。默认不启用 staged gate。
  - p2048：`q4_0/oscar2` staged 约 `2941 tok/s`，`oscar2/oscar2` staged 约
    `2844 tok/s`。
  - pp8192：`q4_0/oscar2` staged 约 `4036 tok/s`，超过 BF16 历史约 `3887`；
    `oscar2/oscar2` staged 约 `3691 tok/s`，接近但略低于 BF16。
  - 8K peak（`runs/staged_fa_peak_20260619T_probe`）：baseline BF16 peak
    `4176 MiB`；`q4_0/oscar2` staged peak `3696 MiB`。速度和显存已同时优于
    BF16，这是目前最接近目标的候选。
  - 质量 10+10（`runs/q4_oscar2_staged_quality_20260619T_probe`）：
    baseline BF16 GPQA/GSM8K `3/10`、`4/10`；`q4_0/oscar2` staged 为
    `3/10`、`2/10`。GPQA 持平，GSM8K 小样本低 20 个百分点，质量还未达最终门槛。
  结论：速度/显存问题基本有候选解；剩余主要是质量。下一步应把 staged FA 与
  BF16 HP recent/sink correction 结合，或提高/修复 V-side 格式质量，而不是再优化
  vec 内层。若要默认化，必须保证 peak 仍低于 BF16，并把 GSM8K 回到接近 BF16。
- Staged FA + HP correction 质量验证（2026-06-19）：`q4_0/oscar2` staged 加
  `LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 LLAMA_KV_HP_PREFILL_ATTENTION=1`
  在 10+10 小集上与 baseline BF16 完全同形：GPQA/GSM8K 均为 `3/10`、`1/10`
  （归档：`runs/q4_oscar2_staged_hp_quality_10x10_20260619T_probe`）。
  这说明 BF16 HP recent/sink correction 能补回此前 staged `q4_0/oscar2`
  的 GSM8K 质量缺口；当前主要问题是 HP 图会掉到慢 fallback，p2048 约
  `1340 tok/s`，远低于无 HP staged 的约 `2941 tok/s`。
- 尝试过把 `K=q4_0,V=oscar2` 接入现有 raw mixed fused vec + HP combine
  （临时 env `LLAMA_KV_MIXED_VEC_RAW_Q4_OSCAR2=1`，已回退）。该路径能命中
  fused op，但 p2048 约 `1212 tok/s`，比慢 fallback 还低，原因是 LP 又回到
  generic vec raw/meta 主循环，丢掉 staged FA 的 tile/MMA 速度优势。不要重复
  这条路线。
- 下一步最有希望的方向变为：让 staged FA LP 路径产出可被 HP combine 直接消费的
  numerator/meta，或实现一个只针对 staged normalized LP + BF16 HP recent 的轻量
  correction kernel。关键是保持 LP 仍走 staged tile/MMA；不要再把 q4/oscar2
  拉回现有 mixed vec 主循环。
- Staged combine 调度/速度复测（2026-06-19）：新增诊断入口
  `LLAMA_KV_HP_STAGED_COMBINE=1`，先让 LP bulk 继续走 staged FA，再用 CUDA
  marker=2 combine kernel 合并 BF16 HP recent/sink。最初 marker=2 未绕过
  `LLAMA_KV_MIXED_VEC_RAW/MAIN` gating，导致该 op 掉到 CPU assert；已修成
  marker=2 直接走 CUDA dispatch。单 query/block combine p2048 约 `1603 tok/s`；
  改成 `ncols=4` warp combine 并补传 Granite `logit_softcap=50` 后，
  `q4_0/oscar2 + HP` p2048 约 `2872 tok/s`，pp8192 约 `2832 tok/s`。
  该 q4K 组合只是速度/质量探针，不是最终 `oscar_int2` 命名。
- 真正 INT2 bulk staged 复测（2026-06-19）：`LLAMA_KV_OSCAR2_ALLOW_STAGED_FA=1`
  下 `oscar2/oscar2` 串行复测 p2048 约 `2501 tok/s`，pp8192 约 `3483 tok/s`；
  加 BF16 HP staged-combine（`HP_SINK=64, HP_RECENT=256`）后 p2048 约
  `1913 tok/s`，pp8192 约 `2437 tok/s`。`HP_RECENT=64` 串行复测 p2048
  约 `2163 tok/s`，pp8192 约 `2540 tok/s`，只小幅改善；靠缩 HP window
  无法达成速度目标。当前 INT2 bulk 已显著快于旧 raw mixed，但仍低于同场 BF16
  pp8192 约 `4176 tok/s`，目标未完成。
- 尝试过优化 OSCAR2 → F16 staging conversion：把一元素一线程转换改成每个
  `block_oscar2_kv` 32 线程解 4 值，构建通过，但纯 `oscar2/oscar2` staged
  约 `2509/3462 tok/s`，带 HP pp8192 约 `2470 tok/s`，与默认基本持平，已回退。
  结论：当前主要瓶颈不是单独的全量 F16 conversion kernel，而是 staged FA 后
  HP correction 的额外扫描/组合，以及 staged 路径仍未直接在 tile 内复用 INT2
  K/V decode。
- 16K 阶梯复测（2026-06-19）：串行 `pp16384` 下 baseline BF16 约
  `3276 tok/s`；纯 `oscar2/oscar2` staged 约 `3020 tok/s`；`oscar2/oscar2`
  staged + BF16 HP combine 约 `2138 tok/s`。长上下文没有自然反超 BF16，
  因此 32K 不是当前调试入口。对照旧 raw vec LP-only pp8192 约 `253 tok/s`，
  再次确认不能回到 generic vec/raw mixed 路线。下一步必须在 staged/tile/MMA
  路径内做 INT2 K/V tile decode 或 exact raw/meta + fused HP correction，而不是
  继续调 raw vec、HP window 或 conversion micro-kernel。
- Staged HP combine mask-skip（2026-06-19）：新增 env
  `LLAMA_KV_HP_STAGED_MASK_SKIP=1`，仅作用于 `LLAMA_KV_HP_STAGED_COMBINE=1`
  的 CUDA marker=2 warp combine；遇到 HP mask `-inf` slot 时跳过 KQ/V 累加。
  串行 A/B：pp2048 `1863 -> 2101 tok/s`，pp8192 `2448 -> 2599 tok/s`，
  pp16384 `2138 -> 2464 tok/s`。这是稳定小正收益，可保留为诊断/候选，
  但仍低于 BF16（8K 约 `4176`、16K 约 `3276`），不能作为目标完成。
  后续不要继续靠 mask/HP window 小调；需要把 HP correction 融进 staged
  raw/meta 或在 staged tile/MMA 内直接做 INT2+HP joint softmax。
- OSCAR2 staging conversion 小优化（2026-06-19）：重新实现并保留
  `dequantize_row_oscar2_kv_f16_cuda` 的默认 blockwise 变体：每个
  `block_oscar2_kv` 使用 `128` 线程、每线程解 1 个值，旧逐元素 256-thread
  kernel 可用 `LLAMA_KV_OSCAR2_DEQUANT_SCALAR=1` 回退。该实验不同于此前
  “32 线程解 4 值”的负收益版本。构建通过；同批 full
  `oscar2/oscar2 + BF16 HP staged-combine + mask-skip` A/B：
  pp2048 default/scalar 约 `2068/2059 tok/s`，pp8192 约 `2535/2499 tok/s`。
  纯 LP staged 也有约 1% 小正收益（pp8192 `3506 -> 3545 tok/s`）。可作为
  默认保留，但幅度太小，目标仍未完成。核心下一步仍是 direct tile-decode
  MMA 或 staged raw/meta + fused HP，不应继续在 conversion micro-kernel
  上投入大量时间。
- Direct tile-decode MMA 原型（2026-06-19）：尝试在
  `fattn-mma-f16.cuh` 中新增 `LLAMA_KV_OSCAR2_DIRECT_MMA=1`，让
  `K=OSCAR2_KV,V=OSCAR2_KV,D=128` 不做全量 F16 staging，而是在 MMA
  tile loader 内把 `block_oscar2_kv` 解到 shared half2，再复用原 staged
  softmax/MMA 主循环。该原型能编译，但为了先跑通关闭了 `cp_async/stream-k`
  路径，结果严重负收益：纯 LP pp2048 `2549 -> 810 tok/s`，pp8192
  `3501 -> 275 tok/s`。已完全撤回 `fattn-mma-f16.cuh` 改动。结论：不能
  通过“把 dequant 塞进通用 MMA loader并禁用 pipeline”解决；若继续 direct
  decode，必须写真正保持 cp_async/stream-k/tile pipeline 的专用 OSCAR2
  kernel，或走 staged raw/meta + fused HP correction。
- Staged combine 固定开销诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_BYPASS=1`，让 marker=2 combine 不扫 HP、不做 softmax，
  只把 `lp_out` device-to-device copy 到最终 `dst`，用于测量 “LP staged graph
  节点 + 额外 combine op/读写” 的下限。已撤回诊断代码。pp8192 normal/bypass
  约 `2471/2687 tok/s`，而同批纯 LP staged 约 `3500 tok/s`。结论：
  HP token 数和 HP scalar KQ 不是唯一主因；即使 combine 计算几乎为零，
  两段 graph/tensor 结构和额外读写仍让 full path 明显低于 LP-only。因此
  marker=2 小修、HP window、mask skip 继续收益有限；下一步必须把 HP correction
  融入 staged LP 输出端，或让 staged LP 原生输出 raw numerator/meta 后在同一
  CUDA op 内完成最终合并。
- Staged fused marker=3 诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_FUSED=1`，让 LP staged FA 直接写最终 `dst`，再原地跑
  HP correction，试图减少 `lp_cur -> marker=2 combine` 中间 tensor/graph
  开销。修正 q/k/v permute 和最终 `reshape_2d` 后可编译运行；pp2048 约
  `2110 tok/s`，pp8192 约 `2601 tok/s`。同场旧
  `LLAMA_KV_HP_STAGED_COMBINE=1` pp8192 约 `2608 tok/s`，撤回后 pp2048
  约 `2127 tok/s`。结论：单纯把 LP 输出写到最终 dst、原地 combine 没有收益，
  已撤回，不要重复 marker=3 / graph in-place 方向。
- Staged raw/meta marker=4 诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_RAW=1`，在 `fattn-mma-f16.cuh` 中尝试让 LP staged FA
  输出 raw numerator + `(max, rowsum)` meta，再复用现有 BF16 HP warp combine，
  以避免 graph split v1 重算 LP KQ/meta。入口、layout、CUDA dispatch 均能跑通，
  但 V1 仍为负收益：固定 `ncols1=4,ncols2=4` 时 pp2048 约 `1341 tok/s`；
  改为复用普通 MMA ncols 选择后 pp2048 单跑约 `1925 tok/s`，pp8192 约
  `1990 tok/s`；同场旧 `LLAMA_KV_HP_STAGED_COMBINE=1` pp2048/pp8192 约
  `1989/2515 tok/s`，撤回后旧路径复测约 `2193/2407 tok/s`。marker=4 代码已
  撤回，源码中不保留 `STAGED_RAW/marker=4/raw_dst_meta` 入口。结论：朴素
  MMA raw/meta V1 没有超过旧 staged-combine，且 raw meta 在 `np>1` 时还会碰到
  per-warp scale/combined max 布局复杂度；后续若再做 raw/meta，必须直接在 staged
  tile 输出端融合 HP correction，或写专用 OSCAR2+HP joint-softmax kernel，不要
  重复 marker=4 两 kernel V1。
- OSCAR2 conversion 64x2 诊断（2026-06-19）：临时加入
  `LLAMA_KV_OSCAR2_DEQUANT_64X2=1`，每个 `block_oscar2_kv` 用 64 线程、每线程
  解 2 个 half，试图在默认 128 线程×1 和已失败的 32 线程×4 之间找平衡。构建
  通过，但 full `oscar2/oscar2 + BF16 HP staged-combine + mask-skip`
  pp2048 从默认约 `2143 tok/s` 降到 `2047 tok/s`，已撤回，不跑 8K。
  结论：OSCAR2→F16 conversion 微调不是当前主瓶颈；不要继续尝试 64x2/32x4
  这类简单线程粒度改写。
- HP active-token 边界诊断（2026-06-19）：不改代码，仅把
  `HP_RECENT=256` 改为 `HP_RECENT=0`，保留 `HP_SINK=64`、
  `LLAMA_KV_HP_STAGED_COMBINE=1`、`LLAMA_KV_HP_STAGED_MASK_SKIP=1`。
  pp2048 约 `2199 tok/s`，pp8192 约 `2660 tok/s`；相比 recent=256 的
  pp8192 `2400-2600` 只有小幅改善，仍远低于纯 LP staged 约 `3500` 和 BF16
  约 `3800-4100`。结论：active HP token 数不是主因，主要瓶颈是两段 graph/
  combine 的结构性读写与额外 kernel；后续不应继续调 HP window，而应实现单 op
  融合或专用 OSCAR2+BF16 HP joint-softmax kernel。
- OSCAR2+HP scalar joint kernel V1（2026-06-19）：临时扩展
  `fattn-oscar2-v2.cu` 的朴素 D=128/4-query OSCAR2 tile kernel，新增
  `LLAMA_KV_MIXED_VEC_OSCAR2_V2_JOINT=1`，让 LP OSCAR2 和 BF16 HP 在同一个
  online softmax kernel 中完成，绕过 staged-combine 的两段 graph/读写。构建可跑，
  但速度很差：pp512 约 `822 tok/s`，pp2048 约 `596 tok/s`，低于当前
  staged-combine 约 `2K+ tok/s`。已撤回，源码无 `OSCAR2_V2_JOINT` 残留。
  结论：单 op 结构本身不够；如果不保持 staged/tile/MMA 级 pipeline，scalar
  joint kernel 会回到旧 raw/mixed vec 的慢区间。下一步必须写真正的 pipelined
  OSCAR2+HP joint kernel，或在 MMA staged tile 输出端原生融合 HP correction。
- V2 残留清理与 staged-combine ncols 诊断（2026-06-19）：复查发现
  `fattn-oscar2-v2.cu/.cuh`、`fattn-vec.cuh` include/launcher env 和 CMake
  引用仍残留在源码中，虽然 joint V1 已判定失败。已删除这些文件与
  `LLAMA_KV_MIXED_VEC_OSCAR2_V2` 入口，`rg` 在 CUDA 源码中无 V2 命中，构建通过。
  清理后 staged-combine 主线可跑：同场 pp2048 约 `2084-2092 tok/s`，pp8192
  单跑有 `2543-2920 tok/s` 波动；BF16 pp8192 同场约 `3426 tok/s`。临时尝试
  `LLAMA_KV_HP_STAGED_COMBINE_NCOLS8=1` 只改变 marker=2 HP combine warp 的
  query 列数：pp2048 有小正收益（约 `2092 -> 2156 tok/s`），但 pp8192 与默认
  几乎相同（约 `2543 tok/s`），不是稳定突破点，实验代码已撤回。结论：当前速度
  已接近但仍低于 BF16；继续做 combine ncols/HP 小修收益有限，主方向仍是把 HP
  correction 融入 staged LP tile/MMA 输出端或实现真正 pipelined OSCAR2+HP
  joint-softmax。
- HP staged-combine softcap 诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_COMBINE_NO_SOFTCAP=1`，只在 marker=2 combine 中忽略
  Granite `logit_softcap=50`，用于判断 scalar HP correction 里的 `tanhf` 是否为
  主要瓶颈。pp2048 默认/禁用 softcap 约 `2111/2023 tok/s`，禁用反而变慢；
  已撤回诊断开关。结论：HP combine 的 softcap 不是主要瓶颈，也不能用跳过 softcap
  做正确路径；继续优化应集中在减少 LP F32 输出读写/额外 combine kernel，或把 HP
  correction 融入 staged LP tile/MMA 输出端。
- HP staged-combine tail-trim 诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_COMBINE_TRIM_TAIL=1`，在 marker=2 combine kernel 内先按
  `hp_kq_mask` 找每个 query 的最后一个 active HP key，再裁剪 HP loop 上限，试图
  避免扫描 padded/未来 HP slot。构建通过；pp2048 默认/trim-tail 约
  `2079/2104 tok/s`，只有约 1% 小正收益；pp8192 默认/trim-tail 约
  `2514/2499 tok/s`，长上下文轻微负收益。已撤回诊断代码。结论：HP mask tail
  扫描不是主瓶颈，额外预扫描开销抵消了少量 key 跳过；不要继续在 HP mask
  range 裁剪上投入，主方向仍是 staged LP 输出端融合 HP correction 或专用
  pipelined joint kernel。
- HP staged FA-add 结构诊断（2026-06-19）：临时加入
  `LLAMA_KV_HP_STAGED_FA_ADD_DIAG=1`，在 staged-combine 分支中让 HP sink+recent
  段单独走现有 BF16 flash-attn/MMA，再把 `lp_cur + hp_cur` 相加。该路径不是正确
  softmax 合并，仅用于判断“用 BF16 FA 处理 HP 是否能绕开 marker=2 scalar combine
  的速度瓶颈”。同场基线：pp2048 LP-only/full/BF16 约 `2597/2105/2824 tok/s`；
  pp8192 LP-only/full/BF16 约 `3451/2535/3588 tok/s`。FA-add 诊断 pp2048 约
  `2391 tok/s`，比正确 full 快但仍低于 LP-only/BF16；pp8192 约 `2572 tok/s`，
  只比正确 full 略快。诊断代码已撤回。结论：scalar HP KQ 有一部分开销，但长上下文
  主要仍是两段 graph/中间 F32 输出读写/额外 combine 结构；正确优化必须把 HP
  correction 融入 staged LP tile/MMA 输出端，或让 staged LP 原生输出可合并的
  raw numerator/meta，不能靠单独 HP FA + 后处理。
- MMA stream-k/meta 审计（2026-06-19）：临时加入
  `LLAMA_KV_MMA_LAUNCH_DEBUG=1` 打印 `launch_fattn` 调度信息，已撤回。OSCAR2 staged
  FA 在 Granite D=128 下走 `ncols=64`、`nbatch_fa=64`、`stream_k=1`，p512 形状
  可见 `blocks=(40,1,1)`、`ntiles_dst=128`、`ntiles_KV=8`，并分配
  `dst_tmp_meta` 做 stream-k fixup。说明 `fattn-mma-f16` 内部已经维护
  `KQ_max/KQ_rowsum` 和 partial numerator，但只作为 scratch/fixup 使用，最终默认
  只把 normalized F32 输出写到 `dst`。下一步如果做正确 raw/meta，应复用/正式暴露
  这套 stream-k meta 和未归一化 numerator，而不是在 graph combine 里重算 LP KQ；
  也不要继续做 graph 层近似相加或 HP 小修。
- MMA raw-numerator fixup 诊断（2026-06-19）：临时加入
  `LLAMA_KV_MMA_RAW_NUMERATOR_DIAG=1`，仅把 stream-k fixup/uniform/general 和
  `flash_attn_combine_results` 的最终除以 rowsum 变成可跳过，用于测试从 MMA
  fixup 尾端输出 numerator 的成本。该诊断不完整：tile 内直接写 `dst` 的路径仍会
  normalized，因此不作为正确 raw/meta 实现。构建通过，但 LP-only pp2048 默认/诊断
  约 `2607/2426 tok/s`，明显负收益，已撤回。结论：只在 fixup/combine 尾端切
  numerator 不够，且会引入模板/分支成本；真正 raw/meta 必须在
  `flash_attn_ext_f16_process_tile` 的 tile 输出阶段统一设计，让直接写和 stream-k
  fixup 使用同一套 raw numerator + meta 布局。
- MMA raw-output 全路径诊断（2026-06-19）：临时加入
  `LLAMA_KV_MMA_RAW_OUTPUT_DIAG=1`，同时在 `flash_attn_ext_f16_process_tile`
  的直接写 `dst` 路径和 stream-k/parallel fixup kernels 中跳过最终 `/ rowsum`，
  让 D=128 MMA staged FA 输出 raw numerator。该路径仍未把 `(max, rowsum)` 暴露为
  graph tensor，因此不是正确 oscar_int2 合并路径，只用于量化 tile 输出端 raw 切口
  成本。构建通过；LP-only pp2048 默认/raw-output 约 `2467/2569 tok/s`，小正收益；
  pp8192 默认/raw-output 约 `3486/3441 tok/s`，长上下文轻微负收益。诊断代码已撤回。
  结论：统一 raw-output 切口可编译运行，但跳过 normalization 本身不是主要瓶颈；
  下一步若继续 raw/meta，必须把 `KQ_max/KQ_rowsum` 作为正式 meta 输出并在同一
  fixup/输出阶段融合 HP correction，而不是只跳过除法。
- MMA LP meta 输出设计审计（2026-06-20）：复查确认 `ggml_flash_attn_ext`
  是单输出 op，`fattn-mma-f16.cuh` 内部的 `KQ_max/KQ_rowsum` 只通过
  `dst_tmp_meta` scratch 参与 stream-k/fixup，默认不会作为 graph tensor 暴露。
  因此不应再尝试 graph-level 重算 LP meta（会抵消 staged FA 收益），也不应把
  LP 输出 tensor 临时扩成 `D+2`（MMA/launcher/fixup 多处假设 `dst->ne[0] == DV`，
  shape 污染风险高）。下一步最小可行实现应是默认关闭的 LP meta sidecar：在
  staged-combine 分支中为 LP FA 分配一个私有 F32 meta tensor，作为 `lp_cur` 和
  HP combine op 的额外 source，CUDA LP FA 在最终输出/fixup 阶段写入真实
  `(max, rowsum)`，HP combine 读取该 sidecar 做精确 online softmax 合并。若该
  sidecar 诊断证明收益/质量成立，再把 HP correction 进一步下沉到 MMA 输出/fixup
  阶段，减少中间 F32 读写和额外 combine kernel。
- LP meta sidecar V1（2026-06-20）：已实现默认关闭诊断
  `LLAMA_KV_HP_STAGED_META=1`。staged-combine 分支为 LP FA 创建私有 F32
  `[2, nq, heads, batch]` meta tensor，并挂到 `lp_cur->src[8]` / combine op
  `src[8]`；CUDA `launch_fattn` 在 stream-k fixup / parallel combine 最终阶段写
  真实 `(KQ_max, KQ_rowsum)`，marker=2 HP graph combine 读取该 sidecar，用
  `lp_out * rowsum` 还原 LP numerator，并从真实 LP max/sum 继续 online softmax。
  默认不开启时原 staged-combine 路径不变。构建通过：
  `cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench`。
  sidecar 2+2 smoke 输出 `2+2 = 4`，无重复退化。单跑 bench（Granite rot-kv，
  HP sink=64/recent=256/staged-combine/mask-skip）：
  - p2048 默认/sidecar 约 `2535/2142 tok/s`，sidecar 有额外 meta 写读开销。
  - p8192 默认/sidecar/sidecar repeat 约 `2562/2852/2961 tok/s`，长上下文有正向信号。
  - 干净 `baseline_bf16` p8192 约 `3874 tok/s`，所以 `oscar_int2` 仍未超过 BF16。
  结论：真实 LP meta 通道已打通，且有长上下文正向信号；下一步不应把 sidecar 作为
  默认终态，而应把 HP correction 融进 MMA stream-k fixup / 输出阶段，消掉 sidecar
  的额外全局写读和 marker=2 combine kernel。
- HP fused-fixup V1（2026-06-20）：新增默认关闭诊断
  `LLAMA_KV_HP_STAGED_FUSED_FIXUP=1`。staged-combine 分支在 LP FA 节点
  `src[5..7]` 挂入 permuted F16 HP K/V 与 F32 HP mask，并直接使用 LP FA 输出，
  不再创建 marker=2 graph combine op。CUDA `launch_fattn` 在 stream-k uniform/
  general fixup 尾端用 LP raw numerator/max/rowsum 扫 HP sink+recent，并写最终
  normalized output；非 D=128 或无 HP source 时保持普通 fixup。该 V1 仍是 scalar
  HP correction，每个 row 一个 D=128 block，用 shared memory 规约 HP KQ，主要用于
  验证“把 HP correction 下沉到 MMA fixup 尾端”能否减少 graph/combine 成本。
  构建通过，`git diff --check` 通过。greedy 2+2 smoke 开头输出 `Q: ... A: 4`，
  无 `2. 2. 2...` 退化；后续 completion 仍会继续生成重复格式文本，需后续用正式
  eval 判断质量。同场 bench（Granite rot-kv，HP sink=64/recent=256）：
  - fused-fixup p2048/p8192 约 `2557/2995 tok/s`。
  - 默认 staged-combine 同批 p2048/p8192 约 `2201/2725 tok/s`。
  - 干净 `baseline_bf16` 同批 p2048/p8192 约 `2776/3590 tok/s`。
  结论：fused-fixup V1 明显优于 marker=2 staged-combine（p8192 约 +10%，p2048
  也恢复到接近 BF16），但仍未超过 BF16。下一步应把 HP correction 从 scalar
  shared-memory loop 改成更轻的 warp/half2/tiled HP correction，或进一步在 MMA
  tile 输出端融合，避免每个 HP key 做 128-thread shared reductions。
- HP fused-fixup warp-reduce 诊断（2026-06-20）：尝试把 V1 HP correction 中每个
  HP key 的 D=128 shared-memory tree reduction 改为 4 个 warp 各自
  `warp_reduce_sum`，再用 warp0 合并并广播 score。第一版未广播到所有线程导致
  correctness 风险；修正广播后构建通过，但 p8192 从约 `2995 tok/s` 掉到约
  `1894 tok/s`，p2048 也仅约 `2590 tok/s`。该改法已回退，构建重新通过，
  回退后 fused-fixup p8192 约 `2845 tok/s`，`git diff --check` 通过。结论：
  简单 warp 分段规约不是突破点，额外同步/广播和更复杂控制流抵消收益；后续若继续
  优化 HP correction，应改成真正的 tiled HP batch 或 half2/vectorized KQ+V，
  不要重复这个 4-warp partial score 方案。
- HP fused-fixup warp0-score 诊断（2026-06-20）：尝试新增默认关闭开关
  `LLAMA_KV_HP_STAGED_FUSED_WARP0_SCORE=1`，只让 warp0 重读 Q/K 并计算完整 D=128
  HP KQ score，再广播给 128 个线程更新 V，目的是减少 shared-memory tree reduction
  的同步次数。构建通过，但同场 A/B 负收益：默认 fused-fixup pp2048/pp8192 约
  `2571/3089 tok/s`，warp0-score 约 `2511/2857 tok/s`。该实验已回退。结论：
  单 warp 串行覆盖 D=128 的 score 计算丢掉了并行度，Q 重读/同步也抵消收益；不要
  重复“单 warp 算完整 HP score”的路线。
- OSCAR2 staged dequant warp4 诊断（2026-06-20）：尝试在
  `dequantize_row_oscar2_kv_f16_cuda` 中新增默认关闭
  `LLAMA_KV_OSCAR2_DEQUANT_WARP4=1`，每个 OSCAR2 D=128 block 用 32 线程、
  每线程解 4 个 INT2 值并用两个 `half2` 写 F16 staging，目标是降低 staged FA
  前置 dequant kernel 开销。构建通过，但 A/B 负收益：默认 fused-fixup
  pp2048/pp8192 约 `2627/2964 tok/s`，warp4 约 `2593/2864 tok/s`。该实验已回退。
  结论：现有 128-thread blockwise staging 虽然朴素，但并行度更好；后续要么避免整段
  staging，要么把 OSCAR2 dequant 下沉到 MMA tile load，而不是把 staging kernel
  简单改成每线程 4 值。
- OSCAR2 staged dequant pairwise 诊断（2026-06-20）：尝试默认关闭
  `LLAMA_KV_OSCAR2_DEQUANT_PAIR=1`，每个 D=128 block 用 64 线程、每线程解 2 值并
  `half2` 写 F16 staging。构建通过；同场 pp2048 默认/pairwise 约 `2551/2495 tok/s`，
  2K 明显负收益；pp8192 单次有一次 `2925/3017` 的噪声正收益，但复测为
  `2970/2967`，基本持平。该实验已回退。结论：staging kernel 的 half2/线程粒度小修
  没有稳定收益；速度差距应继续从“避免整段 staging”或“OSCAR2 direct MMA tile-load”
  解决。
- MMA sync-load 诊断（2026-06-20）：临时加入
  `LLAMA_KV_MMA_FORCE_SYNC_LOAD=1`，在 host 端把
  `ggml_cuda_fattn_mma_get_nstages()` 强制返回 0，用来估算 direct OSCAR2 tile-load
  第一版如果不能走 cp_async/multistage 会损失多少。构建通过，但 pp2048 运行直接
  CUDA error（backtrace 到 CUDA add op 后 abort），说明当前 MMA shared-memory
  layout/launch attribute 与简单 host 端改 nstages 不兼容。该实验已回退。结论：
  direct OSCAR2 MMA tile-load 不能靠运行时强制同步加载来低风险插入，必须作为专用
  D=128 kernel/template 路径设计，显式处理 shared memory layout 和 pipeline。
- OSCAR2 direct MMA tile-load V1（2026-06-20）：实现过默认关闭
  `LLAMA_KV_OSCAR2_MMA_DIRECT=1` 诊断，只在 D=128、K/V 都是 `GGML_TYPE_OSCAR2_KV`
  时跳过整段 K/V→F16 staging，并在 MMA tile load 阶段把 OSCAR2 block 直接解到
  shared `half2` tile。第一版为了降低复杂度强制 direct path 使用同步 shared load
  结构，不接 cp_async/multistage。构建通过；pp2048 默认/direct 约 `2552/2187 tok/s`，
  pp8192 默认/direct 约 `2812/2182 tok/s`，明显负收益。该实验已回退。结论：
  “只把 dequant 搬进 MMA tile”不足以超过 staged F16 路径，原因是失去 cp_async
  pipeline、tile 内解码开销和额外控制流抵消了省掉全量 staging 的收益。后续若继续
  direct 路线，必须做 packed/multistage direct loader，或把 OSCAR2 tile decode
  与 cp_async 风格的双缓冲 pipeline 等价融合；不要保留同步 direct loader。
- HP fused-fixup half2-score 诊断（2026-06-20）：尝试在
  `flash_attn_hp_fused_finish_row` 中新增默认关闭
  `LLAMA_KV_HP_STAGED_FUSED_HALF2_SCORE=1`，让 HP KQ score 用 64 个线程按
  `half2` 成对读取 K 并计算两个维度，V 更新仍保持 128 线程完整写 D=128。构建通过，
  2+2 smoke 正常，但同场 pp2048 默认/half2-score 约 `2641/2586 tok/s`，tg64
  也从约 `68.5` 掉到 `60.1`，负收益。该实验已回退且源码无
  `HALF2_SCORE/half2_score` 残留。结论：HP score 这段不能靠简单 half2 成对加载解决；
  寄存器/Q 重读/分支成本抵消了减少 load 的收益。
- LP meta sidecar combine 复测（2026-06-20）：在当前 staged/fused 优化后重新测试
  已有 `LLAMA_KV_HP_STAGED_META=1` 路径（不开 `LLAMA_KV_HP_STAGED_FUSED_FIXUP`），
  即 LP FA 写真实 sidecar meta，再由 marker=2 standalone warp combine 合并 HP。
  pp2048 约 `2246 tok/s`，仍明显慢于 fused-fixup 同批约 `2600+ tok/s`。结论：
  standalone HP combine 的额外 kernel/global 读写当前仍不划算；短期默认应继续使用
  fused-fixup，下一步若想超过 BF16，需要做真正 tile 级 HP correction 或 packed/
  multistage OSCAR2 loader，而不是回到 sidecar combine。
- HP fused-fixup recent-gate 诊断（2026-06-20）：尝试默认关闭
  `LLAMA_KV_HP_STAGED_FUSED_RECENT_GATE=1`，根据 `LLAMA_KV_HP_SINK` 和 query col
  粗略裁掉 recent HP 槽，想减少 fused HP correction 中被 mask 掉的 HP key 扫描。
  构建通过，短 bench 正常；pp2048 default/gate 约 `2544/2581 tok/s` 有小正收益，
  pp8192 第一组 default/gate 约 `2940/2992 tok/s` 也仅小幅正收益，但复测 gate 掉到
  `1789 tok/s`。该实验已回退。结论：HP recent ring 的有效性不能只用 query col 和
  `ne01 - recent_slots` 推断，缓存位置/slot 语义更复杂，错误跳过 recent 会带来正确性
  和性能风险；不要重复这种粗粒度 recent 裁剪。若要裁 HP，必须基于真实 HP mask/slot
  元数据做 tile-level compact/scan。
- HP fused-fixup warp-tile V1 诊断（2026-06-20）：确认纯 OSCAR2 staged LP bulk
  pp8192 约 `3664 tok/s`，已经接近 `baseline_bf16` 约 `3733 tok/s`；加 BF16 HP
  fused correction 后回落到约 `2921-3006 tok/s`，主要瓶颈集中在 HP correction 合并。
  尝试默认关闭 `LLAMA_KV_HP_STAGED_FUSED_WARP_TILE=1`，把已有 standalone graph combine
  的 one-warp-per-row HP 算法直接移入 stream-k uniform fixup，避免额外 combine kernel。
  构建通过；pp512 default/warp-tile 约 `2277/2455 tok/s`，pp2048 约 `2530/2580 tok/s`
  有小正收益，但 pp8192 约 `2957/2845 tok/s` 负收益。该实验已回退。结论：
  one-warp-per-row HP 算法在短 prompt 能降低 shared reduction 成本，但塞进 stream-k
  fixup 后长上下文 occupancy/block 形状/partial combine 成本更差；不要按这个 V1
  形态默认化。下一步若继续 HP 优化，应做真正 tile-level HP correction（多个 query
  共享 HP K/V tile 和 softmax状态）或基于真实 HP mask/slot metadata compact，而不是
  单纯把 row 内规约换成 warp 写法。
- OSCAR2 staged fused-fixup 正确性纠偏（2026-06-20）：重新检查 graph/CUDA dispatch
  后发现，`LLAMA_KV_OSCAR2_ALLOW_STAGED_FA=1` + `LLAMA_KV_HP_STAGED_FUSED_FIXUP=1`
  下的 OSCAR2 LP FA 会选择普通 vec 后端，而该 vec 后端不消费挂在 `src[5..7]` 的
  BF16 HP K/V/mask。因此此前 staged fused-fixup pp2048 约 `2.5-2.7k`、
  pp8192 约 `2.9-3.0k` 的数字属于 LP staged FA 伪快路径，不是完整
  `oscar_int2 = BF16 HP + INT2 bulk`。已在 `llama-graph.cpp` 禁止 OSCAR2 走该
  staged fused-fixup 伪路径，并在请求该 env 时直接落回正确 raw mixed marker=1
  CUDA combine。修复后同样 env 下 pp2048 约 `975 tok/s`，2+2 smoke 输出正常
  （`2+2=4...`，无 `2. 2. 2...`）。后续优化基线必须使用完整 raw mixed combine，
  不能再把 staged LP-only 数字作为 oscar_int2 速度。
- OSCAR2 staged+meta 正确路径更新（2026-06-20）：修复 marker=2 CUDA support 条件，
  允许 `LLAMA_KV_MIXED_VEC_RAW=1` 下的 staged graph combine 使用 F32 LP mask；同时
  OSCAR2 staged combine 默认写 LP sidecar meta，让 HP combine 用真实 LP `(max,sum)`
  合并，而不是重算 LP KQ 或走 LP-only 伪路径。当前完整 `oscar_int2` 推荐诊断 env：
  `LLAMA_KV_OSCAR2_ALLOW_STAGED_FA=1 LLAMA_KV_HP_STAGED_COMBINE=1
  LLAMA_KV_HP_STAGED_MASK_SKIP=1`，且不要设置 `LLAMA_KV_HP_STAGED_FUSED_FIXUP=1`。
  同场 `-p8192 -n64` 多次复测：`oscar_int2` 常见好点约 `2.7-2.9k tok/s`，低谷可到
  `1.5k`；`baseline_bf16` 正常约 `3.8k tok/s`，也偶有低谷约 `1.8k`。因此速度已有
  明显进展，但按稳定正常值仍未超过 BF16。`-p2048 -n64` 当前约 `2.2-2.4k tok/s`。
- HP staged combine 诊断（2026-06-20）：尝试在 standalone marker=2 warp combine 中
  默认关闭地把 HP K/V 每个 key staged 到 shared，目标是让 ncols 个 query warp 共享
  BF16 HP K/V load。构建通过，但 pp2048 从约 `2205 tok/s` 掉到约 `634 tok/s`，说明
  每 key 两次 `__syncthreads()` 的代价远大于减少 global read 的收益。该实验已撤回，
  源码无 `LLAMA_KV_HP_STAGED_COMBINE_STAGE_KV` / `stage_hp_kv` 残留。不要重复这种
  per-key block-wide sync staging。
- HP recent 成本诊断（2026-06-20）：在完整 staged+meta 路径上，`HP_RECENT=256`
  时 pp8192 n64 约 `2871 tok/s`；`HP_RECENT=128/64` 分别约 `3353/3336 tok/s`。
  pp2048 n64 从 recent256 约 `2256` 提到 recent128/64/0 的约 `2368/2408/2420`。
  这确认当前主要额外成本在 BF16 HP correction 扫描/合并，而不是 LP OSCAR2 bulk。
  但 recent 不能直接降为默认；必须先跑 GPQA/GSM8K 等质量集确认 BF16 HP 窗口缩小
  不会导致相对 BF16 超过 5-10% 的准确率下降。
- CLI smoke 注意事项（2026-06-20）：当前 `llama-cli` 直接用 Granite base 模型跑
  `What is 2+2?` 时，`oscar2/oscar2` 和 `bf16/bf16` 都可能在 120-180s timeout 前
  重复输出 `>`，因此这条 CLI prompt 不能作为区分 oscar_int2 correctness 的可靠判据。
  速度 correctness 先以 `llama-bench` 能正常完成为准；文本质量需要改用现有
  GPQA/GSM8K harness 或修正 Granite base 的 prompt/template 后再测。
- HP mixed PPL correctness 诊断（2026-06-20）：用固定短 corpus 扩展版
  `/tmp/oscar_int2_quality_smoke_corpus_long.txt` 跑 `llama-perplexity -c 512 --chunks 2`。
  `baseline_bf16` 正常，PPL 约 `1.009`；纯 `oscar2/oscar2` 无 HP 正常，PPL 约
  `1.10`。开启 HP prefill 后发现：
  - raw mixed `sink64/recent0` 原本 NaN；修复 HP combine 中 LP meta/output 非有限值处理后，
    PPL 约 `1.208`，不再 NaN。
  - `sink0/recent64` 不 NaN，但 PPL 约 `7.92`，说明 recent-only 质量仍明显差。
  - staged+meta marker=2 在 finite guard 后不再 NaN，但 PPL 仍约 `83.5`，质量不可用。
  - `LLAMA_KV_HP_NO_EXCLUDE=1` 只是诊断：fused raw sink64 可从 NaN 变为 PPL 约 `1.30`，
    但 staged+meta PPL 约 `45.7` 且 pp8192 约 `1392 tok/s`，不能作为默认或最终方案。
  结论：当前可作为 correctness 基线的是 raw mixed safe path，但速度慢
  （pp2048 约 `991 tok/s`，pp8192 约 `290 tok/s`）。staged+meta 速度快
  （pp8192 约 `2.6k tok/s`），但 LP sidecar/meta 合并语义仍不正确，不能作为最终
  oscar_int2。下一步必须修 staged LP meta/HP combine 的数学语义，把 raw mixed 的质量
  和 staged LP 的速度合并，而不是继续只追 staged+meta 速度。
- staged+meta 失败分支补充（2026-06-20）：尝试过两个快速定位修复，均已撤回：
  - marker=2 graph combine 改用未 permute 的 Q。结果 PPL 从约 `83.5` 进一步恶化到约
    `157458`，说明该 kernel 仍期待当前 permuted Q 布局；不要重复这个改法。
  - graph combine 忽略 LP sidecar meta、把 normalized LP output 当作 `(max=0,sum=1)`
    初始化。PPL 只从约 `83.5` 降到约 `67.5`，仍远不可用，说明问题不只是 sidecar
    meta 未初始化或量级错误，而是 staged+meta 的 LP/HP joint-softmax 语义整体不等价。
  保留的 correctness 修复只有 raw/mixed combine 中对 LP meta/output 非有限值和空分母
  的防御；它能让 raw mixed sink64 从 NaN 恢复到 PPL 约 `1.208`，但速度仍慢。下一步
  应把 raw mixed 的正确 joint-softmax 结构做 tiled/vectorized，而不是继续试
  staged+meta 的近似合并。
- raw mixed LP KV_max 诊断（2026-06-20）：尝试把普通 FA 的
  `flash_attn_mask_to_KV_max` 预处理直接接到 raw mixed LP（env
  `LLAMA_KV_MIXED_VEC_RAW_KVMAX=1`，已撤回）。LP-only pp2048 从约 `992` 到
  `1085 tok/s`，pp8192 从约 `307` 到 `337 tok/s`，但 full pp8192 仍约
  `291 tok/s`，且 PPL smoke 从 raw safe 的约 `1.2` 恶化到约 `5.43`。原因是
  raw mixed 的 LP self-mask 会额外把 BF16 HP sink/recent 位置置为 `-inf` 以避免
  double count；直接从该 mask 反推 KV 上界会把仍需由 HP 分支参与 joint softmax 的
  token 错误裁掉。因此不要再复用当前 self-mask 做 LP KV_max。若要做正确裁剪，必须
  从 graph/kv-cache 侧传入不含 HP-exclusion 的纯 causal/SWA bound sidecar，或在
  mixed op 中接收额外 mask/bounds；这属于 op schema/graph 级改动。
- raw mixed bound-sidecar KV_max 诊断（2026-06-20）：随后实现过 env-gated
  `LLAMA_KV_MIXED_VEC_RAW_BOUND_KVMAX=1`，在 graph 侧额外创建“不排除 HP”的
  F16 causal/SWA mask，挂到 mixed op `src[9]`，CUDA raw LP 用该 sidecar 生成
  KV_max，LP 分数仍用原始 HP-exclusion mask。该版本能 build，PPL 与当时 default
  recent256 一致（约 `5.43`，没有额外恶化），但速度无收益：pp2048 从约 `999`
  降到约 `943 tok/s`，pp8192 与 default 同为约 `291 tok/s`。该实验已全部撤回，
  源码无 `RAW_BOUND_KVMAX` / `mask_lp_bound` / `self_kq_mask_bound` 残留。结论：
  长上下文当前慢点不是可由 coarse KV_max sidecar 解决的未来-key 扫描；不要继续扩大
  mask/bounds sidecar 方向，除非 profiler 证明 LP key 扫描重新成为主瓶颈。
- staged LP final_meta 写入诊断（2026-06-20）：尝试让普通 FA 在
  `parallel_blocks==1` 且 `src[8]` 存在时直接写 LP `(max,sum)` sidecar，并让
  `fattn-vec` 在 `dst_meta != nullptr` 时也写 meta，目的是确认 staged+meta 的
  PPL 83 是否由 LP meta 未初始化导致。构建通过，但 staged+meta PPL 仍约 `83.5`，
  无质量收益；同时 raw mixed pp2048 从约 `~1000` 掉到约 `530 tok/s`，明显负收益。
  该实验已撤回。结论：staged+meta 质量问题不是简单的 final_meta 未写入；不要把
  `dst_meta` 写入扩展到普通 `parallel_blocks==1` vec FA 热路径。
- staged LP FA 节点挂载诊断（2026-06-20）：进一步发现原 staged 分支把
  `lp_meta` 挂在 `build_attn_mha()` 返回的 2D reshape tensor 上，而不是实际
  `GGML_OP_FLASH_ATTN_EXT` 节点。临时展开 LP FA，改为把 sidecar 挂到原始 FA 节点并用
  原始 4D `lp_fa` 做 combine，PPL 从约 `83.5` 降到约 `37.8`，但仍远不可用；
  `sink64/recent0` 更差，约 `1244`。纯 `oscar2/oscar2` FA 同一 corpus PPL 约
  `1.10`，raw mixed `sink64/recent0` 约 `1.208`，说明 OSCAR2 FA 本体是正常的，
  问题在 staged LP 输出/meta 暴露给 HP combine 的接口语义，而不是量化格式本身。
  该临时 graph 改动已撤回；后续若继续 staged 路线，必须设计一个正式的
  “FA 同时返回 normalized 2D output + per-row `(max,sum)` meta”的接口，而不是
  手工把 sidecar 挂到 reshape/view 或在普通 vec 热路径强行写 meta。
- staged LP output/layout 追加诊断（2026-06-20）：实现过最小正式接口尝试：
  新增临时 `build_attn_mha_fa_with_raw()`，让 staged 分支同时拿到 reshape 后 LP output
  和原始 FA 节点/permute 后 Q/K；`lp_meta` 按 `q_perm` 的 `(token, head, stream)`
  尺寸创建，并挂到原始 FA 节点。CUDA marker=2 combine 也临时支持 2D
  `lp_out=(D*head, token)` 索引。构建通过，但 PPL 仍为约 `37.8`，无质量收益。
  进一步的 `LLAMA_KV_HP_STAGED_LP_COPY=1` 诊断（只把 LP output 复制到 dst，不扫 HP）
  得到 NaN，说明 staged graph 中 LP output/sidecar 作为 combine 输入仍不可靠；
  问题不只是 2D/4D 索引。该接口尝试和 LP-copy/shape debug 已撤回。后续不要再把
  staged+meta 当主优化路线；应回到 raw mixed 正确路径，把已正确的 joint-softmax
  CUDA kernel 做 tiled/vectorized 优化。
- raw mixed 护栏复测（2026-06-20）：撤回 staged 接口尝试后重新验证可用基线。
  `LLAMA_KV_MIXED_VEC_RAW=1 LLAMA_KV_MIXED_VEC_MAIN=1 HP_SINK=64 HP_RECENT=0`
  下，短 corpus PPL 约 `1.2077`，说明 raw joint-softmax correctness 仍可用；
  但速度仍远未达标，pp2048 约 `926.6 tok/s`，pp8192 约 `281.5 tok/s`。
  这组数字是后续 raw-kernel 优化的当前护栏：不能牺牲 PPL 正确性去追 staged 高速假象。
- raw mixed two-tier/HP combine 诊断（2026-06-20）：针对上述 raw 护栏继续排查。
  - 尝试 `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_STRIDE=2`，`TAIL=128/256`，`MODE=end`。
    PPL smoke 直接恶化到约 `137` / `250`，质量不可用；不要把 LP token 采样作为
    `oscar_int2` 默认加速路线，除非另做正式格式/质量分支。
  - 保留一个正确性等价的小优化：给 generic `flash_attn_ext_vec` 增加编译期
    `raw_oscar2_two_tier` 参数。默认 raw mixed 不再把 two-tier 采样状态和分支编进
    OSCAR2/OSCAR2 LP kernel；只有显式 two-tier env 才走兼容实例。构建通过，PPL
    smoke 约 `1.192`（同一小样本波动内，无退化），pp2048 约 `1220-1234 tok/s`。
    8K LP-only 诊断恢复到约 `695-699 tok/s`，说明此前部分慢点来自默认实例携带
    two-tier 运行时开销/寄存器压力。
  - full `oscar_int2` pp8192 仍只有约 `405-475 tok/s`，显著低于 LP-only
    `~695 tok/s`；下一步瓶颈应看 HP combine/完整 graph 调度/输出重写，而不是继续
    盲目改 LP KQ/V 热循环。
  - 试过在 HP combine kernel 内按 `hp_sink` + `hp_recent_tail` 只扫两个短区间，
    语义 smoke 通过（PPL 约 `1.192`），但 pp8192 full 仍约 `410 tok/s`，无收益；
    该 range 裁剪已撤回。说明 full 掉速不是简单由 HP 中间大段 masked token 的 KQ/V
    计算造成；不要继续沿 HP range 裁剪方向小修。
  - `HP_SINK=0/HP_RECENT=0` 诊断下 full pp8192 仍约 `251 tok/s`，没有接近 LP-only。
    说明 graph 里 HP tensor 形状仍不是空段，只是 mask 屏蔽；临时 `K_hp->ne[1]==0`
    早退不会命中，已撤回。若要让 no-HP 真正跳过 combine，必须在 graph 构建侧直接
    bypass HP combine 或把 HP 段长度作为显式运行时参数传入 kernel。
  - 尝试过把 `mctx_cur->get_n_hp_kv()` 作为 `ggml_flash_attn_ext_mixed` 的额外
    op param 传给 CUDA，并用它限制 HP combine 的 `ne11_hp`。构建与 PPL smoke
    正常，`HP_SINK=64` debug 显示 `hp_active=64`，但 sink64 本来未 padding，pp8192
    仍约 `421 tok/s`；`HP_RECENT=64` pp2048 也仅约 `1235 tok/s`，无明确收益。
    该 API/schema 改动已撤回，源码无 `n_hp_kv_active` / `hp_active` /
    `ne11_hp_active` 残留。结论：full path 的主要问题不是 padded HP 长度，而是
    第二阶段 HP combine/输出重写本身；后续应考虑把 HP correction 融进 LP raw kernel
    或在 graph 侧真正 bypass 无 HP combine，而不是继续传长度小修。
  - 临时 `LLAMA_KV_MIXED_VEC_HP_COPY_LP_ONLY=1` 诊断（已撤回）：raw LP 正常写
    normalized output/meta 后直接返回，跳过 HP combine。pp8192 单次约 `554 tok/s`；
    同场 LP-only 约 `580 tok/s`，full sink64 约 `479 tok/s`。结论：第二阶段 HP
    combine correction 约有 10-15% 8K 成本，但即使完全跳过也离 BF16 很远；
    因此下一步应把 HP correction 融进 LP kernel 以省掉第二 pass，同时继续优化
    LP raw kernel 本体，而不是继续做独立 combine 小修。
  - HP combine mask 读取诊断（2026-06-20）：实现过临时
    `LLAMA_KV_MIXED_VEC_HP_NOMASK=1`。完全跳过 HP mask 会把早期 query 的未来
    sink key 混入，PPL 从约 `1.192` 变到约 `1.159`，不是等价路径；修正为
    `col >= ne11_hp` 后才把 mask 当 0，PPL 回到约 `1.192`，但速度无收益。
    同批 pp2048 default/nomask 约 `1213/1218` 与 `1224/1249 tok/s`，pp8192
    default/nomask 约 `480/475` 与 `409/408 tok/s`。该诊断已撤回，源码无
    `HP_NOMASK` 残留。结论：HP mask load/branch 不是当前 8K 主要瓶颈，不要继续
    沿独立 HP combine mask 微调；若要减少 10-15% combine 成本，应做 LP+HP fused
    correction 或 graph-level bypass，而不是仅优化 mask 分支。
  - 默认路径小清理（2026-06-20）：two-tier LP 采样是失败分支。默认 raw mixed
    已不再每次把默认 `fattn_vec_oscar2_two_tier` 参数异步拷贝到 device symbol；
    只有显式 `LLAMA_KV_MIXED_VEC_LP_TWO_TIER_STRIDE>1` 时才拷贝。PPL smoke 仍约
    `1.192`，pp2048/pp8192 无稳定速度收益（约 `1208/412 tok/s` 单次），该改动
    作为清理保留，不应被记录为性能突破。
  - constant-param fused HP correction 诊断（2026-06-20）：尝试过
    `LLAMA_KV_MIXED_VEC_FUSED_HP=1`，通过 CUDA constant 参数块把 HP K/V/mask 指针
    传给 OSCAR2 raw LP 模板实例，并在 LP final write 阶段直接合并 HP correction，
    以跳过第二个 `hp_combine_warp` kernel。构建通过，但 correctness 不达标：
    默认 CUDA graphs 下 PPL 第二个 chunk 出 NaN；尝试关闭 graphs 后 NaN 消失但 PPL
    仍约 `1.179`，没有对齐默认约 `1.192`。该实验已撤回，源码无
    `FUSED_HP` / `fattn_vec_mixed_hp` / `raw_oscar2_fused_hp` 残留。结论：
    不能用 device constant 指针偷传 HP 参数，也不能把 HP correction 粗暴塞进
    generic LP final-write block-wide reduce。若继续 fused 路线，必须做正式专用 kernel
    参数接口，并先用逐 token/chunk PPL 对齐默认 joint-softmax 语义。
  - HP combine contiguous-dim 诊断（2026-06-20）：尝试过
    `LLAMA_KV_MIXED_VEC_HP_CONTIG=1`，把 `hp_combine_warp` 中每个 lane 处理的维度从
    `lane, lane+32, lane+64, lane+96` 改为连续 `4*lane..4*lane+3`，希望提升
    HP K/V 和 LP numerator 的读取局部性。构建通过，但 PPL smoke 从默认约 `1.192`
    变为约 `1.173`，不满足等价 correctness；该分支已撤回，源码无 `HP_CONTIG` /
    `contiguous_dims` 残留。结论：HP combine 的维度分片不能简单重排；warp 内求和和
    输出维度布局需要保持严格一致。不要继续做独立 HP combine 布局小修，除非同时有逐元素
    输出 diff 验证。
  - OSCAR2 KQ float LUT 诊断（2026-06-20）：尝试把 generic raw LP 中
    `GGML_TYPE_OSCAR2_KV` 的 shared `Q*centroid` LUT 从 `half` 改成单独的 `float`
    LUT，避免热循环里反复 `__half2float(turbo_lut[...])`。构建通过，PPL smoke 约
    `1.1899`（接近默认 `1.192`），但速度无稳定收益：full pp2048/pp8192 约
    `1225/480 tok/s`，8K 与默认约 `478 tok/s` 持平；LP-only pp8192 约 `585 tok/s`，
    低于此前常见 `~700 tok/s`。该改动已撤回。结论：增大 shared memory 换 float LUT
    不是主方向；LP 仍需要真正 tiled/vectorized 主循环，而不是继续改 KQ LUT 精度。
  - OSCAR2 LP V-tile v1 诊断（2026-06-20）：尝试在 `fattn-vec.cuh` 内直接新增
    env-gated `LLAMA_KV_MIXED_VEC_LP_VTILE_V1`，把 OSCAR2 V 按 KV tile 批量 decode
    到 shared 后复用。第一版未进入有效 bench：ncols=4 时 ptxas shared memory
    约 `0xcc00`，ncols=8 时约 `0x11800`，均超过 RTX 5050 当前 `0xc000` 上限；随后
    试图把 V tile 缩到 64 key 时，patch 误污染旧 mixed/dedicated kernel 的循环边界。
    该实验已完整撤回，源码无 `LP_VTILE` / `vtile_keys` / `V_tile` / `launch_generic_`
    残留；构建、2+2 smoke 通过，回归 pp2048 约 `1151 tok/s`、pp8192 约
    `468 tok/s`。结论：V-tile 方向若继续，必须做干净隔离的专用 D=128 kernel
    或新 `.cu/.cuh`，并先把 shared memory 预算压到 ncols=4 可编译范围内；不要继续
    在 generic `fattn-vec.cuh` 主路径中硬塞大 shared tile。
  - HP combine shared-stage 诊断（2026-06-20）：在 `hp_combine_warp` 中试过
    env-gated `LLAMA_KV_MIXED_VEC_HP_STAGE=1`，把最多 64 个 BF16 HP K/V staging 到
    shared，让同一个 block 的 4 个 query warp 复用。构建通过，PPL smoke 对齐默认
    `1.1920`，但 pp8192 只有约 `412.6 tok/s`，低于默认约 `467.9 tok/s`；该实验
    已撤回，源码无 `HP_STAGE` / `stage_hp` / `hp_K_stage` / `hp_V_stage` 残留。结论：
    HP K/V global load 不是 second-pass combine 的主要瓶颈，shared staging 的同步和
    occupancy 成本更高；后续若要消掉 full path 与 LP-only 的差距，应减少/融合第二个
    combine pass，而不是在独立 combine kernel 里 staging HP K/V。
  - fused HP v2 诊断（2026-06-20）：尝试通过现有 `flash_attn_ext_vec` 的 `sinks`
    参数传 device-side HP 参数块，在 raw LP final-write 阶段直接合并 HP correction，
    以省掉 standalone `hp_combine_warp`。构建通过，但 PPL smoke 第一 chunk 已偏到
    约 `1.1742`，随后 CUDA illegal memory access；该实验已撤回，源码无
    `FUSED_HP2` / `fused_hp` / `oscar2_fused_hp` 残留。结论：generic `fattn-vec`
    final-write block reduction 不是安全的 HP fusion 插入点；要融合 second-pass
    combine，必须写独立、清晰的专用 fused kernel（例如一 warp 负责一个 query 的
    LP finalize + HP correction），不要复用 `sinks` 指针或 CUDA constant 指针偷传参数。
  - second-pass copy-only 诊断（2026-06-20）：新增临时
    `LLAMA_KV_MIXED_VEC_COPY_LP_DIAG=1`，让 LP raw 正常写 normalized output/meta，
    第二个 kernel 只把 LP output 原样 copy 回 dst，不做 HP correction。pp8192 约
    `708.7 tok/s`，几乎等于 LP-only 约 `704.8 tok/s`，明显快于 full 约
    `467.9 tok/s`。结论：第二个 kernel 的 launch/读写下限不是主要问题；full path
    的主要成本在 HP correction 的 KQ/online-softmax/V 算术本身。后续应优化 HP
    correction 算术结构或减少必须精确 correction 的 HP key，而不是只做 graph-level
    bypass 或 copy/write 小修。该开关仅作诊断，不是质量正确路径。
  - HP linear-softcap 诊断（2026-06-20）：试过临时
    `LLAMA_KV_MIXED_VEC_HP_LINEAR_SOFTCAP=1`，在 HP correction 中用线性近似替代
    `logit_softcap*tanhf(score)` 以估计 tanh 成本上限。pp8192 约 `469.8 tok/s`，
    与 full 默认同一量级，远低于 copy-only `~708.7 tok/s`；该实验已撤回，源码无
    `HP_LINEAR_SOFTCAP` 残留。结论：HP correction 慢点不是 tanh 本身，而是 64 个
    HP key 的 KQ/online-softmax/V 全链路算术与 per-query warp 组织。
  - HP FA split 诊断（2026-06-20）：尝试 `LLAMA_KV_MIXED_VEC_HP_FA_SPLIT=1`，
    让 LP 继续走 OSCAR2 raw normalized，HP 另走现有 F16/F16 `flash_attn_ext_vec`
    raw normalized 快路径，再用 tiny kernel 按 `(max,sum)` 合并两个 normalized
    分布。构建通过，但 PPL smoke 直接出现 `nan`；该实验已撤回，源码无
    `HP_FA_SPLIT` / `combine_two_norm` 残留。结论：普通 F16/F16 FA 的 raw output/meta
    不能直接替代当前 HP correction，至少 mask/causal/meta 语义未对齐；若继续 split
    路线，必须先做逐 token/逐 head output+meta diff 工具，而不是直接把两套 FA 结果相加。
  - HP_FAST 专用化诊断（2026-06-20）：试过 `LLAMA_KV_MIXED_VEC_HP_FAST=1`，
    针对当前默认 `normalize_lp && parallel_blocks=1 && mask_skip=false &&
    logit_softcap!=0` 写了一个去掉 runtime 分支/part loop 的 HP combine kernel。
    加 LP meta valid guard 后 PPL smoke 约 `1.1975`，但 pp8192 只有约 `409 tok/s`，
    与 full 默认同量级且远低于 copy-only `~708.7 tok/s`；该实验已撤回，源码无
    `HP_FAST` / `hp_combine_fast` 残留。结论：简单手写专用化不能解决 HP correction
    成本，必须真正改变并行组织，例如让多个 query 共享 HP K/V tile 或做逐元素 diff 后
    正确接入更高效的 HP FA/meta 路径。
  - HP mask-skip 默认打开（2026-06-20）：把 `hp_combine_warp` 的
    `LLAMA_KV_MIXED_VEC_HP_MASK_SKIP` 默认值从 off 改为 on，仍可用
    `LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=0` 回到旧行为。构建通过；2+2 smoke 输出
    `The answer is 4.`；PPL smoke 约 `1.1920`，未出现 NaN。当前串行 bench：
    pp2048 约 `1218.5 tok/s`，pp8192 约 `460.7 tok/s`，比同轮旧默认 pp8192
    约 `409.8 tok/s` 有小幅恢复，但仍远低于 LP-only/copy-only 约 `686/695 tok/s`
    和 BF16 约 `3.7k+ tok/s`。结论：该改动可保留为低风险默认，但它只是减少 HP
    combine 的无效 mask 区域工作；最终目标仍必须优化 LP raw 主循环与 HP correction
    并行组织，不能把 mask-skip 当主突破方向。
  - HP mask all-valid fast path（2026-06-20）：在默认 mask-skip 路径里，若当前
    query 的 HP mask 首尾均为 0 且整段未被裁剪，则 HP correction loop 内直接使用
    `mask_val=0`，避免 64 个 HP sink key 上反复读取 F32 mask。构建通过；PPL smoke
    仍约 `1.1920`；pp2048 约 `1232.4 tok/s`，pp8192 约 `478.3 tok/s`。这是小幅
    正收益并可保留，但仍未接近 8K `>=700 tok/s` gate，更远未追上 BF16。该结果再次
    说明 HP correction 的主要成本是 KQ/softmax/V 算术与 warp 组织，不是单纯 mask load。
  - HP ncols=8 combine 复测（2026-06-20）：临时让 `LLAMA_KV_MIXED_VEC_HP_NCOLS8=1`
    在 mask-skip 默认打开时也能命中。pp8192 约 `430 tok/s`，只在默认同轮 `413 tok/s`
    波动附近，没有明确收益；该诊断入口已撤回，源码不保留 `HP_NCOLS8` 路径。不要继续
    做 HP combine ncols 小调作为主方向。
  - staged combine correctness 修复（2026-06-21）：修复了
    `LLAMA_KV_HP_STAGED_COMBINE=1` 的两个实质 correctness bug：
    1) LP FA 的 `lp_meta` sidecar 必须在 `build_attn_mha()` 内、FA 节点 `cb()` 前挂到
       `cur->src[8]`，否则 CUDA launcher 看到 `final_meta=0`，combine 读到 `(0,0)`。
    2) staged combine 也必须像普通 FA/raw mixed 一样在 `logit_softcap != 0` 时先
       `scale /= logit_softcap`，否则 Granite `softcap=50` 下 HP KQ 分数被放大约 50x。
    修复后小 PPL smoke 恢复到同量级：raw sink64/recent0 约 `7.3928`，
    staged sink1/recent0 约 `7.6917`，staged sink64/recent0 约 `6.8055`。
    但速度是负收益：pp2048 staged/raw 约 `711.7/1247.3 tok/s`，
    pp8192 staged/raw 约 `245.5/444.1 tok/s`。因此 staged combine 只能作为显式
    correctness/实验路径保留，不能作为默认速度路线。
  - raw mixed 当前诊断（2026-06-21）：同机 BF16 pp8192 约 `3616.6 tok/s`；
    raw oscar_int2 pp8192 约 `420-444 tok/s`，LP-only 约 `621.8 tok/s`。
    `HP_RECENT=0/64/256` 都在同一量级，`LLAMA_KV_MIXED_VEC_HP_MASK_SKIP=0`
    约 `428 tok/s`，说明 HP window/mask 仍不是主瓶颈。即使完全跳过 HP correction，
    LP-only 也远低于 BF16，下一步必须优化 OSCAR2 LP bulk 主循环（专用 D=128 tiled
    KQ/V/online-softmax 或更高效 raw-meta FA），而不是继续 staged graph、HP window、
    mask skip 或独立 combine 小修。
  - raw mixed 小幅保留优化（2026-06-21）：generic `flash_attn_ext_vec` 中
    `GGML_TYPE_OSCAR2_KV` 的 `nthreads_KQ` 从 2 调为 1。构建通过；PPL smoke
    保持 `6.0836 +/- 0.45919`；pp2048 full 约 `1262-1316 tok/s`，pp8192
    LP-only 约 `718.9 tok/s`，full 约 `484.8-495.4 tok/s`。这是小幅正收益，
    但没有改变与 BF16 的数量级差距。
  - HP compact mask 小幅保留优化（2026-06-21）：raw HP combine 在默认
    sink+recent compact HP cache 下，用 `hp_sink` 和 LP effective length 直接计算
    当前 query 的有效 HP key 范围，避免每个 query 线性扫描 HP mask 首尾。可用
    `LLAMA_KV_MIXED_VEC_HP_COMPACT_MASK=0` 回退旧行为。A/B：pp8192 compact on
    约 `495.8 tok/s`，off 约 `488.0 tok/s`；PPL smoke 不变。该优化可保留，
    但只是减少 mask 处理开销，不能作为主突破方向。
  - HP sink64 unroll 诊断（2026-06-21）：尝试在 compact mask 下对最常见的
    `key_end=64` sink-only case 加固定 64 次 unroll，结果 pp8192 从约 `495.8`
    降到约 `441.9 tok/s`，PPL 仍正常但速度负收益；该实验已撤回。不要继续在
    HP combine warp loop 上做大 unroll/寄存器展开，小幅分支消除不抵指令体积和
    occupancy 损失。
  - HP 64-lane combine 诊断（2026-06-21）：尝试 `LLAMA_KV_MIXED_VEC_HP_64LANE=1`，
    每个 query 用 64 个线程处理 D=128（每线程 2 维）以减少 per-thread KQ/V 工作。
    构建通过，但 pp8192 约 `434.6 tok/s`，低于同轮默认约 `505.2 tok/s`；该实验已撤回，
    源码无 `HP_64LANE` / `64lane` 残留。结论：跨两个 warp 合并 score 的 shared/sync
    与 occupancy 成本高于减少单线程维度负载；不要继续沿“更多线程处理单 query”的方向微调。
  - 当前可靠瓶颈（2026-06-21）：`LLAMA_KV_MIXED_VEC_COPY_LP_DIAG=1` 下 pp8192
    约 `727.7 tok/s`，full 约 `495 tok/s`；第二个 kernel 的纯读写/copy 不贵，
    掉速来自 HP correction 的 BF16 KQ/online-softmax/V 算术。后续要冲 8K
    `>=700 tok/s`，需要改变 HP correction 的并行组织（例如让多个 query 共享 HP
    K/V tile，或正确接入高效 HP FA/meta combine），同时继续提升 OSCAR2 LP bulk；
    不应再重复 HP window、mask scan、64-key unroll、CUDA graph 或 staged combine。
  - HP key 数量质量/速度诊断（2026-06-21）：简单减少 HP sink/recent 不能作为默认
    提速方案。同一 smoke 里 `sink/recent=64/256` pp8192 约 `442.5 tok/s`、PPL
    `6.0836`；`32/128` pp8192 约 `485.0 tok/s` 但 PPL `271.8`；`16/64`
    pp8192 约 `473.9 tok/s`、PPL `61.0`；`0/0` pp8192 约 `253.0 tok/s`、PPL
    `7.6010`。结论：质量依赖足够的 BF16 HP correction，且速度并不随 HP key
    数线性改善；不要把缩 HP window 当成达标方案。
  - HP FA split2 诊断（2026-06-21）：新增临时 marker=3 combine-two 路径，让 LP
    和 HP 分别走 `ggml_flash_attn_ext` 产出 normalized output + `(max,sum)` meta，
    再用 tiny kernel 合并两个分布。PPL smoke 正确（`6.0836 +/- 0.45919`），证明
    normalized FA+meta 数学合并语义可行；但速度明显负收益：split2/default
    pp2048 约 `791/1289 tok/s`，pp8192 约 `252/502 tok/s`。该实验已撤回，源码无
    `SPLIT2` / `split2` / `combine_two` / `marker=3` 残留。结论：当前图形态下双 FA
    节点成本过高，不能替代手写 HP correction；若要复用 BF16 FA，必须更深地融合
    LP raw output 和 HP FA/meta，避免额外完整 LP staged FA 和 graph overhead。
  - HP combine fast exp 小幅保留优化（2026-06-21）：将 raw
    `flash_attn_ext_mixed_oscar2_hp_combine_warp` 中 online softmax 的
    `expf` 改为 CUDA `__expf`，只作用于 HP correction warp kernel，不改 LP FA。
    PPL smoke 保持 `6.0836 +/- 0.45919`；pp2048 约 `1317 tok/s`，pp8192 约
    `494.5 tok/s`，与当前默认波动上沿同档、无质量回退。该改动可保留为小优化，
    但距离 BF16 仍有数量级差距，不能替代后续 fused/tiled 主循环工作。
  - HP mask all-valid loop 专门化复测（2026-06-21）：尝试把 compact mask 下
    `hp_mask_all_valid` 的 HP correction loop 拆成专门无 mask-load 分支，避免每个
    key 检查 mask。PPL smoke 仍为 `6.0836 +/- 0.45919`，但 pp2048 约
    `1261.6 tok/s`、pp8192 约 `490.5 tok/s`，没有超过 fast-exp 默认路径，且代码
    复杂度更高；该实验已撤回。当前源码只保留普通 loop 中的
    `hp_mask_all_valid ? 0.0f : mask read` 判断。
  - HP correction 分段诊断（2026-06-21）：临时加入 `LLAMA_KV_MIXED_VEC_HP_DIAG`
    拆分 HP combine 成本。默认同轮 pp8192 约 `455.1 tok/s`；`no_v`（保留 HP
    KQ/softmax，但跳过 V 累加）约 `502.6 tok/s`；`v_only`（跳过 HP KQ，保留
    softmax/V 形式）约 `434.9 tok/s`。结论：HP V 累加有明显成本，但 KQ/softmax
    warp 组织也仍是瓶颈，不能只靠移除/近似 V 累加达标。该诊断开关已撤回。
  - HP softcap fast-tanh 诊断（2026-06-21）：临时用 `__expf` 形式计算 HP
    correction 的 `tanh` softcap。PPL smoke 不变（`6.0836 +/- 0.45919`），但
    pp8192 `485.8 tok/s`，低于同轮默认 `488.0 tok/s`；该诊断已撤回。softcap
    的 `tanhf` 不是当前主要突破口。
  - 当前 clean 验证点（2026-06-21）：撤回 `HP_DIAG` / `HP_FAST_TANH` 后重新 build
    通过，PPL smoke `6.0836 +/- 0.45919`，默认 raw mixed pp2048 约
    `1327.3 tok/s`，pp8192 约 `505.7 tok/s`，tg64 约 `67.3 tok/s`。目标仍未完成：
    速度仍远低于 BF16，后续要继续做真正的 OSCAR2 LP tiled 主循环或 HP correction
    跨 query/tile 复用，而不是 softcap/mask/window 小修。
  - HP qtile4 combine 候选（2026-06-21）：新增 env-gated
    `LLAMA_KV_MIXED_VEC_HP_QTILE=1`，只覆盖 raw mixed 默认形态（D=128、ncols=4、
    normalized LP、parallel_blocks=1）。一个 warp 同时处理 4 个 query 的 HP
    correction，在同一个 HP key 上复用 K/V load，保持 LP meta/online-softmax 语义。
    PPL smoke 保持 `6.0836 +/- 0.45919`。同轮 A/B：default/qtile pp2048
    约 `1321.5/1456.1 tok/s`，pp8192 约 `489.9/694.6 tok/s`；后续 8K repeat
    为 `613.8/694.8/698.6 tok/s`，追加同批 default/qtile 为 `445.8/613.4 tok/s`。
    结论：qtile4 是目前第一条结构性正收益 HP combine 路径，明显快于默认，但仍未稳定
    超过 8K `>=700 tok/s` gate，更远未追上 BF16；暂时保留为 env-gated 候选，不改默认。
    下一步应在 qtile4 内继续优化 K/V load、mask/key_end 计算与寄存器压力，或结合 LP
    tiled 主循环继续提升。
  - HP qtile4 key_end 预计算（2026-06-21）：将 compact HP mask 下每个 query 的
    `key_end` 从 key loop 内提到 qtile 初始化阶段，避免每个 HP key 重复计算
    `sink_valid/recent_valid`。PPL smoke 仍为 `6.0836 +/- 0.45919`；qtile pp2048
    约 `1526.8 tok/s`，pp8192 约 `666.8 tok/s`，后续回退负收益实验后单跑
    pp8192 约 `702.6 tok/s`。该改动保留在 env-gated qtile 候选路径内；8K
    已有过线单点但仍有明显波动，未达到稳定超过 700 gate，更未接近 BF16。
  - HP qtile4 默认候选（2026-06-21）：将 raw mixed 默认形态（normalized LP、
    `parallel_blocks=1`、`ncols=4`）下的 HP qtile4 设为默认开启，保留
    `LLAMA_KV_MIXED_VEC_HP_QTILE=0` 作为 opt-out 回退。默认 PPL smoke 仍为
    `6.0836 +/- 0.45919`；默认 pp2048/pp8192 约 `1499.2/703.6 tok/s`；
    opt-out pp8192 约 `498.5 tok/s`。这是当前默认 `oscar_int2` 的主要速度进展，
    但仍远低于 BF16；后续瓶颈仍是 OSCAR2 LP bulk。
  - HP qtile4 负收益小修（2026-06-21）：尝试把同一个 HP key 的 K 向量读入
    `k[4]` 后供 4 个 query 复用，PPL 正常但 pp2048/pp8192 降到约
    `1396.0/608.3 tok/s`，大概率是寄存器压力抵消了 K load 减少，已撤回。
    又尝试把 compact mask 分支做成 qtile 模板常量，PPL 正常、pp2048 约
    `1528.8 tok/s`，但 pp8192 约 `602.8 tok/s`，也已撤回。不要继续做这两类
    qtile 微调；当前最佳 qtile 形态是“4 query/warp + key_end 预计算”，K 仍在
    每个 query 分支内直接读。
  - qtile 后瓶颈复测（2026-06-21）：`LLAMA_KV_MIXED_VEC_HP_QTILE=1` 下 pp8192
    full / LP-only / copy-LP 分别约 `695.4 / 720.4 / 716.7 tok/s`。HP qtile
    correction 只剩约 3-4% 拖累，当前主要瓶颈已回到 OSCAR2 LP bulk 主循环本身。
    后续继续优化 HP combine 收益有限，应转向 LP tiled/vectorized 主循环。
  - LP softmax fast-exp 诊断（2026-06-21）：尝试仅在 OSCAR2/OSCAR2 LP vec
    softmax 中把 `expf` 换为 `__expf`。PPL smoke 不变（`6.0836 +/- 0.45919`），
    LP-only pp8192 仍约 `721.5 tok/s`，但 qtile full pp8192 降到约 `595.3 tok/s`；
    该改动已撤回。结论：LP 瓶颈不是简单 `expf`，不要继续做函数替换级小修。
  - LP KQ norm-hoist 诊断（2026-06-21）：尝试在 OSCAR2 KQ LUT 路径中先累加
    centroid LUT 和，再将 K row 的 `norm` 乘法移出 D 内层循环。PPL smoke 不变，
    但 qtile full pp8192 降到约 `577.9 tok/s`，LP-only pp8192 降到约
    `627.4 tok/s`；该改动已撤回。结论：当前 generic LP vec 对寄存器/指令调度
    极敏感，继续做 KQ 内层代数小重排风险高、收益低，应转向真正专用 tiled 主循环。
  - LP-only ncols sweep（2026-06-21）：在默认 qtile 已开启但设置
    `LLAMA_KV_MIXED_VEC_RAW_LP_ONLY_DIAG=1` 下测试 `LLAMA_KV_MIXED_VEC_NCOLS=1/2/4/8`，
    pp8192 分别约 `718.7 / 725.0 / 738.3 / 536.4 tok/s`。结论：LP bulk 当前最佳
    仍是 ncols=4，ncols=8 明显变慢，因此不应做 HP qtile8；后续应继续做 LP
    D=128 专用 tiled 主循环，而不是扩 query tile 宽度。
  - 当前 8K 对照（2026-06-21）：默认 qtile 开启后，同轮 `baseline_bf16` pp8192
    约 `3786.8 tok/s`，默认 `oscar_int2` pp8192 约 `729.9 tok/s`。`oscar_int2`
    已稳定超过 700 gate，但仍只有 BF16 约 19%，距离“速度优于 BF16”目标很远。
    后续必须继续做 OSCAR2 LP bulk 专用 tiled/vectorized 主循环。
  - OSCAR2 KQ float LUT 诊断（2026-06-21）：尝试在 generic LP vec 的 OSCAR2 KQ
    路径额外使用 shared `float` centroid LUT，避免内层 `__half2float(turbo_lut)`。
    PPL smoke 不变，但默认 full pp8192 约 `696.3 tok/s`，LP-only pp8192 约
    `714.9 tok/s`，低于当前默认/LP-only 上沿；该改动已撤回。结论：shared memory
    膨胀/occupancy 成本高于 half 转换收益，不应继续沿 float LUT 微调。
  - OSCAR2 KQ 8维展开诊断（2026-06-21）：尝试把 generic LP KQ 内层从每 4 维
    解一次 `qs/rs` 改成每 8 维解一次，以减少循环次数和 `rs` 读取。PPL smoke 不变，
    但默认 full pp8192 约 `583.4 tok/s`，LP-only pp8192 约 `597.3 tok/s`，明显
    低于当前默认；该改动已撤回。结论：继续展开 generic vec KQ 会显著增加寄存器/指令
    体积，不能作为 LP 提速路线。
  - OSCAR2 KQ reduce-skip 小清理（2026-06-21）：generic LP vec 中 OSCAR2 KQ 已固定
    `nthreads_KQ=1`，因此显式跳过 `warp_reduce_sum<nthreads_KQ>`。PPL smoke 不变；
    pp8192 首跑 full/LP-only 约 `706.9/724.0 tok/s`，repeat 约 `702.2/720.0 tok/s`。
    该改动保留为语义等价的小清理，但收益在波动范围内，不是突破方向。
  - LP no-V 诊断（2026-06-21）：新增临时 env `LLAMA_KV_MIXED_VEC_LP_DIAG_NO_V=1`
    让 generic raw LP 仍计算 KQ/online softmax/meta，但跳过 OSCAR2 V decode 与
    `prob * V` 累加。pp8192 同轮 default full 约 `696-725 tok/s`，LP-only 约
    `714-719 tok/s`，LP no-V 约 `1013-1396 tok/s`（波动较大但明显更快）。结论：
    当前最大剩余块确实是 V decode/累加；后续应做真正的 V tile/staging 或
    `prob tile x V tile` 批量化，而不是继续调 HP combine。
  - OSCAR2 V m/d hoist 诊断（2026-06-21）：尝试在 V hot loop 外按 key hoist
    `block_oscar2_kv` 指针与 `m/d` half2，再手写 centroid pair FMA。结果 full
    pp8192 约 `720.7 tok/s`，LP-only pp8192 约 `600.9 tok/s`，低于默认 LP-only
    约 `714.4 tok/s`；该改动已撤回。结论：简单 hoist 会增加寄存器/调度压力，
    不能替代真正的 V tile 复用。
  - OSCAR2 V rows/thread=8（2026-06-21）：将 generic raw LP 中 OSCAR2 V 的
    `V_rows_per_thread` 从 4 调为 8，减少 V hot loop 迭代次数。构建通过，PPL smoke
    仍为 `6.0836 +/- 0.45919`；pp8192 full 单次约 `756.7 tok/s`，LP-only 约
    `794.7 tok/s`，复测 pp2048/pp8192 full 约 `1477.9/763.2 tok/s`。该改动
    先保留；它确认 V 粒度仍有可用优化空间，但距离 BF16 pp8192 约 `3786.8 tok/s`
    仍差很远。
    后续已收紧为仅对 `GGML_TYPE_OSCAR2_KV` 且 `D == 128` 使用 rows/thread=8，
    避免影响非目标 D=64/oscar2 sanity 实例；复测 pp2048/pp8192 full 约
    `1467.8/760.7 tok/s`，PPL smoke 仍为 `6.0836 +/- 0.45919`。
  - OSCAR2 V rows/thread=16 诊断（2026-06-21）：尝试继续把 `V_rows_per_thread`
    从 8 提到 16，编译期触发 `ggml_cuda_memcpy_1<nbytes=32>` 的 `bad nbytes`
    static assert（D=64 oscar2/oscar2 template instance）。已回退到 8；除非重写
    combine/shared 写回，不要简单再把该粒度提到 16。
  - OSCAR2 inline rows=8 V decode 诊断（2026-06-21）：尝试让 generic vec 的
    OSCAR2 专用 V inline 分支也覆盖 `V_rows_per_thread=8`，一次解两个 packed byte。
    full pp2048/pp8192 单次约 `1528.0/767.5 tok/s`，但 LP-only pp8192 掉到
    `672.1 tok/s`，低于 generic rows=8 的约 `794.7 tok/s`；已撤回。结论：
    rows=8 保留，但不要在 generic vec 里继续手写更重的 inline decode，寄存器/调度
    成本会吃掉收益。
  - 旧 dedicated LP rows=8 诊断（2026-06-21）：把
    `flash_attn_ext_mixed_oscar2_lp_raw_vec` 的 `V_rows_per_thread` 从 4 临时改到 8。
    编译通过，但 pp8192 full/LP-only 仅约 `452.5/454.5 tok/s`，远低于 generic
    rows=8 的约 `760/795 tok/s`；已回退。结论：旧 dedicated LP kernel 的整体
    KQ/V/accumulator 结构仍不适合作为下一步基础，真正专用 kernel 需要重写主循环。
  - OSCAR2 `dequantize_V_oscar2<half, ne=8>` 手写专用化诊断（2026-06-21）：
    尝试在 `fattn-common.cuh` 中为当前 generic rows=8 快路径手写一次 `m/d` half2
    复用、两个 packed byte 解码。构建通过，但 pp2048/pp8192 full 约
    `1445.9/650.4 tok/s`，LP-only pp8192 约 `655.3 tok/s`，低于 generic
    rows=8；已撤回。结论：helper 内手写展开同样会伤编译器调度/寄存器布局，
    后续不应继续在 dequant helper 层做小修。
  - 当前 rows=8 下 OSCAR2 V lane 粒度复测（2026-06-21）：把 generic vec 中
    OSCAR2 `nthreads_V` 从 `nthreads_V_q/2`（D=128 时 16 lanes）临时降到
    `nthreads_V_q/4`（8 lanes）。构建通过，但 pp2048/pp8192 full 约
    `1312.3/466.4 tok/s`，LP-only pp8192 约 `487.5 tok/s`，明显低于 16-lane
    rows=8；已回退。结论：当前 generic 快路径保持 rows/thread=8 + 16 V lanes。
  - OSCAR2 direct-prob V 累加（2026-06-21）：在 generic raw LP 主循环中，对
    `GGML_TYPE_OSCAR2_KV/GGML_TYPE_OSCAR2_KV` 且 `nthreads_KQ=1` 的路径，用
    `__shfl_sync` 直接把当前 warp 中的 `KQ_reg[j]` softmax 概率广播给 V lanes，
    避免 V hot loop 从 shared `KQ[...]` 回读概率。构建通过，PPL smoke 仍为
    `6.0836 +/- 0.45919`；首轮 full pp2048/pp8192 约 `657.7/770.7 tok/s`
    （2K 明显为波动），复测 full pp2048/pp8192 约 `1481.7/765.1 tok/s`，
    LP-only pp8192 约 `811.5 tok/s`。该改动先保留；收益不大但方向正确，
    后续可继续考虑跳过 shared `KQ` 写入或进一步专用化 LP 主循环。
  - direct-prob skip shared KQ store 诊断（2026-06-21）：在上述 direct-prob
    路径下尝试进一步跳过每轮 `KQ[j*nthreads + tid] = KQ_reg[j]` 的 shared 写入。
    构建通过，但 pp2048/pp8192 full 约 `1503.4/639.1 tok/s`，LP-only pp8192
    约 `779.6 tok/s`，低于保留版 direct-prob；已撤回。结论：shared KQ 写入
    虽然不再被 OSCAR2 V hot loop 读取，但保留它对调度/寄存器/后续路径更稳。
  - direct-prob 后 ncols 复测（2026-06-21）：在当前 direct-prob + rows=8
    路径下，用 `LLAMA_KV_MIXED_VEC_RAW_LP_ONLY_DIAG=1` 复测 pp8192 LP-only：
    `ncols=2/4/8` 分别约 `588.2/818.4/469.0 tok/s`。结论：ncols=4 仍是当前
    最佳点，不应切到 2 或 8。
  - `LLAMA_KV_MIXED_VEC_V_ACCUM_V1` 诊断（2026-06-21）：尝试在当前 direct-prob
    上增加 env-gated OSCAR2 V accum v1 分支，对 rows=8 一次手写解两个 4-value
    pack。构建通过，但同批 pp8192 LP-only default/v1 约 `807.8/663.0 tok/s`，
    明显负收益；已撤回。结论：这种“8-value inline decode”仍不是需要的 V tile，
    会伤寄存器/调度，不应继续沿此方向小修。
  - direct-prob 跳过 `__syncwarp()` 诊断（2026-06-21）：由于 OSCAR2 direct-prob
    路径不再从 shared `KQ[...]` 读概率，尝试跳过 V loop 前的 per-iteration
    `__syncwarp()`。构建通过，但 pp2048/pp8192 full 约 `1512.7/657.8 tok/s`，
    LP-only pp8192 约 `645.1 tok/s`，低于保留路径；已撤回。结论：该同步点仍
    对调度/可见性/编译器行为有帮助，不应移除。
  - OSCAR2 KQ LUT stride 诊断（2026-06-21）：尝试将 OSCAR2 shared centroid LUT
    第三维 stride 从 generic 的 `8+1=9` 改成 `8`，减少 shared footprint。PPL smoke
    不变，但 full pp8192 约 `610.9 tok/s`，LP-only pp8192 约 `703.4 tok/s`，低于当前
    默认；该改动已撤回。结论：`+1` padding 对 bank conflict/调度仍有价值，不要移除。
  - 旧 dedicated LP 同构化诊断（2026-06-22）：把
    `flash_attn_ext_mixed_oscar2_lp_raw_vec` 临时改到更接近当前 generic 快路径：
    `nthreads_KQ=1`、V rows/thread=8、half2 accumulator、direct-prob shuffle。
    构建通过，但同批 pp8192 LP-only default/dedicated 约 `750.5/484.4 tok/s`。
    已撤回。结论：旧 dedicated kernel 的 shared/规约/寄存器组织本身不适合继续修，
    不是简单对齐 rows/half2/direct-prob 就能追上 generic。
  - 2026-06-22 no-V 复测：当前 default LP-only pp8192 约 `752.2 tok/s`，
    `LLAMA_KV_MIXED_VEC_LP_DIAG_NO_V=1` 约 `1313.7 tok/s`。V decode/`prob*V`
    仍然是最大剩余块，但需要真正的 V tile/staging 或主循环重写；helper/inline
    小修已经反复负收益。
  - OSCAR2 V 显式 `__hfma2` 诊断（2026-06-22）：在 generic half2 V hot loop 中
    对 `GGML_TYPE_OSCAR2_KV` 临时把 `tmp*KQ + VKQ` 改成显式 `__hfma2`。构建通过，
    但 pp8192 LP-only/full 约 `749.4/730.0 tok/s`，与默认同批 `752/725` 级别
    基本持平，无明确收益；已撤回。结论：编译器已能处理该 half2 累加，不应继续
    做指令级 FMA 微调。
  - OSCAR2 V coefficient accumulation 诊断（2026-06-22）：尝试 env-gated
    `LLAMA_KV_MIXED_VEC_RAW_V_COEFF=1`，利用 `V = m + d * centroid(idx)` 将
    `prob*m` 作为 per-query base 单独累加，热循环只累 `prob*d*centroid`，最后
    写回时把 base 加回每个维度。构建通过，2+2 smoke 输出 `2+2 is 4`，但同批
    pp8192 LP-only default/v_coeff 为 `751.01/750.99 tok/s`，完全无收益；已撤回。
    结论：mean 项拆分不是当前瓶颈，下一步必须改变 V tile/staging/主循环粒度，
    不是继续改 V 表达式。
  - 2026-06-22 当前默认诊断复测：pp8192 LP-only 约 `752.4 tok/s`，LP no-V
    约 `1309.2 tok/s`，full `oscar_int2` 约 `728.5 tok/s`。HP combine 掉速很小，
    最大空间仍在 LP V decode/`prob*V`，但 generic vec 内的小修已基本耗尽。
  - OSCAR2 shared V staging 诊断（2026-06-22）：尝试 env-gated
    `LLAMA_KV_MIXED_VEC_RAW_V_STAGE=1`，在 generic raw LP 的每个 warp 内先把
    32-key × D=128 的 OSCAR2 V 解码到 shared `half2` tile，再用现有 probability
    累加路径读取 shared tile。构建通过，但同批 pp8192 LP-only default/stage 为
    `752.10/750.42 tok/s`，略慢；已撤回，源码无 `RAW_V_STAGE` / `raw_oscar2_v_stage`
    / `V_stage` 残留。结论：把 V staging 塞进现有 generic vec 主循环不能获得复用收益，
    下一步若继续追速度，需要换 kernel 形态/输出组织，而不是在当前 loop 周围加 shared staging。
  - v2 独立落点恢复（2026-06-22）：重新新增 `fattn-oscar2-v2.cu/.cuh`，并在
    `CMakeLists.txt` 中编译 `fattn-oscar2-v2.cu`。入口由
    `LLAMA_KV_MIXED_VEC_OSCAR2_V2=1` 触发，目前只做参数检查/debug 打印并返回 `false`
    回退 generic raw LP，不改变默认行为。构建通过；p512 debug 能看到
    `mixed_oscar2_v2: stub q=512 k=512 ncols=4 softcap=50 fallback=1`。无 debug
    p512 LP-only default/v2-fallback 约 `1790.5/1748.0 tok/s`，属于 fallback 波动范围。
    结论：后续真正专用 D=128 kernel 应在该独立文件继续实现，避免继续污染
    `fattn-vec.cuh` generic 主循环。
  - v2 入口下沉 + warp4 V1（2026-06-22）：将 v2 hook 从 `fattn.cu` 外层 dispatch
    移到 raw LP launch 层，新增
    `ggml_cuda_flash_attn_ext_mixed_oscar2_v2_lp_try(...)`，可直接拿到
    `ne11_lp_eff`、`parallel_blocks`、`normalize_lp`、`lp_meta/lp_num`、softcap 等
    完整上下文。新增 force-only 真 kernel：
    `LLAMA_KV_MIXED_VEC_OSCAR2_V2=1 LLAMA_KV_MIXED_VEC_OSCAR2_V2_FORCE=1`，
    仅覆盖 `D=128/ncols=4/parallel_blocks=1/normalized LP`，一个 block 内 4 warp
    分别处理 4 个 query。构建通过，p512 LP-only smoke 正常命中
    `mixed_oscar2_v2_lp: force=1 kernel=warp4`，无 CUDA 报错。但性能明显低于
    generic raw：同批 LP-only pp2048 default/v2 为 `1563.9/883.0 tok/s`，
    pp8192 default/v2 为 `751.0/251.1 tok/s`。默认 full pp2048 仍约
    `1438 tok/s`，force-only 代码不影响默认路径。结论：warp-per-query V1 只作为
    独立 v2 写回/正确性骨架保留，不应继续微调其内层；下一步必须实现真正的
    KV tile/query-tile 复用版本，否则比 generic vec 组织差太多。
  - v2 `tile4_shared_v` 诊断（2026-06-22）：在 v2 force-only 下新增
    `LLAMA_KV_MIXED_VEC_OSCAR2_V2_TILE=1`，一个 block 覆盖 4 query，每个 key 先
    计算 4 个 query 的 KQ/prob，再由 warp0 将该 key 的 D=128 OSCAR2 V 解码到 shared，
    4 个 query 复用这份 V。构建通过，p512 debug/smoke 命中
    `kernel=tile4_shared_v` 且无 CUDA 报错，但性能更差：p512 LP-only 约
    `1080 tok/s`；同批 pp2048 default/tile 为 `1516.6/695.3 tok/s`，
    pp8192 default/tile 为 `754.8/191.4 tok/s`。结论：每 key 做 block-wide
    `__syncthreads()` 的 shared V staging 成本远大于 V decode 复用收益；不要继续
    沿 per-key shared staging 修。若继续 v2 tile，必须改成 KV tile 内多个 key 批量
    staging/score 后再同步，或者重新设计 warp/register 级 V 复用，避免 key 级全 block
    同步。
  - OSCAR2 raw fast-exp 诊断（2026-06-22）：先误把
    `LLAMA_KV_MIXED_VEC_RAW_FAST_EXP` 的 `getenv()` 放进 device kernel，导致 CUDA
    编译失败；已撤回并恢复 build。随后改成 constexpr 只对 OSCAR2 raw 路径使用
    `__expf`，构建通过，但 pp2048 full 从约 `1447 tok/s` 掉到约 `1402 tok/s`，
    且有 softmax 数值风险；已撤回。结论：不要在 device 内读 env，也不要把 fast
    exp 当作当前主优化方向。
  - 2026-06-22 恢复后默认复测：build 通过，full `oscar_int2` pp2048 约
    `1447.5 tok/s`、pp8192 约 `729.3 tok/s`；正确 no-V 开关是
    `LLAMA_KV_MIXED_VEC_LP_DIAG_NO_V=1`，不是旧的 `RAW_DIAG_NO_V` 名字。同批
    LP-only pp8192 约 `747.1 tok/s`，LP no-V 约 `1326.8 tok/s`。V 累加仍是最大
    剩余块，但在 generic loop 内的小修收益已经很小。
  - v2 KV-batch shared V staging 诊断（2026-06-22）：尝试新增 force-only
    `LLAMA_KV_MIXED_VEC_OSCAR2_V2_TILE4=1`，每 4 个 key 批量计算 KQ/prob，
    再将 4×D 的 V 解码到 shared，试图把同步成本从每 key 摊到每 4 key。构建通过，
    但 pp2048 LP-only default/tile4 约 `1492.2/733.1 tok/s`，仍明显慢于 generic。
    该代码已撤回，build 再次通过。结论：无论 per-key 还是小 key-batch 的 block-level
    shared V staging 都不适合当前 GPU/布局；后续应转向 warp/register 级复用或改变
    LP 输出组织，而不是继续加 shared V staging 变体。
  - 2026-06-22 CUDA 资源/参数复核：`cuobjdump --dump-resource-usage` 显示当前
    generic OSCAR2 raw `flash_attn_ext_vec<128,4,oscar2,oscar2,...>` 约
    `REG 197-199 / SHARED 15360`，v2 `warp4` 仅 `REG 56 / SHARED 0` 但仍慢很多，
    说明问题不是单纯寄存器数，而是算法组织/内存复用形态。HP combine `QTILE=1`
    是正收益：同批 full pp8192 `QTILE=1/0` 约 `725.5/493.4 tok/s`，默认应继续开启。
    ncols 复核：pp2048 `ncols=1/2/4/8` 约 `1408/1438/1449/1247 tok/s`，pp8192
    `ncols=2/4` 约 `731.7/733.9 tok/s`；`ncols=4` 仍是默认最佳点，继续调 ncols
    没有突破空间。
  - LP two-tier 速度上限诊断（2026-06-22）：不改默认，仅通过 env 测试远端 LP
    降采样。当前 default full pp2048/pp8192 约 `1430/732 tok/s`。`stride=4,tail=512,
    weighted=1` 得到 pp2048/pp8192 约 `1375/896 tok/s`；`stride=8,tail=512,weighted=1`
    得到约 `1651/1030 tok/s`，2+2 smoke 正常输出 `4`。但该路径改变 LP attention
    语义，质量未验证，且速度仍远低于 BF16 8K 约 3.8k tok/s；不能作为默认或目标完成。
    结论：减少远端 LP 工作量能显著提速，但要达成目标仍需质量验证与更系统的 LP
    压缩/采样设计，或继续重写 exact LP kernel。
  - LP two-tier 质量 smoke（2026-06-22）：使用 `llama-completion` 跑 3+3
    GPQA/GSM8K，对比 `baseline_bf16`、`oscar_int2_current`、以及通过
    `--extra-env` 覆盖到 `stride=8,tail=512,weighted=1,mode=end` 的
    `oscar2_int2_mixed_vec_twotier`。结果归档：
    `runs/oscar_int2_twotier_stride8_quality_smoke_20260622T120207/summary.md`。
    小样本结果：BF16 GPQA/GSM8K `1/3,1/3`；default oscar_int2 `1/3,0/3`；
    two-tier `1/3,0/3`。样本太小不能下最终准确率结论，但 two-tier 没显示出比
    default 更好，且当前 oscar_int2 在 GSM8K smoke 已落后 BF16。结论：two-tier
    只能作为速度/召回折中候选，不能作为“质量接近 BF16”的默认修复。
  - HP recent 质量/速度复核（2026-06-22）：扫 `LLAMA_KV_HP_RECENT=256/512/1024/
    2048/4096`，full pp2048 约 `1419/1535/2151/2217/2321 tok/s`，pp8192 约
    `731/754/797/896/1171 tok/s`。recent 增大反而提速，原因是 LP effective
    length 变短且 HP combine qtile 路径足够快；但 8K 仍远低于 BF16 约 3.8k tok/s。
    对 `HP_RECENT=4096` 跑 10+10 GPQA/GSM8K 质量 smoke，结果归档：
    `runs/oscar_int2_hprecent4096_quality_10x10_20260622T120945/summary.md`。
    结果与 BF16 完全一致：BF16 GPQA/GSM8K `2/10,4/10`，oscar_int2_current
    `2/10,4/10`。结论：质量问题在该小样本上可通过更大 BF16 HP recent 解决；
    下一步应以 `HP_RECENT=4096` 作为质量候选，继续优化速度/显存，而不是使用
    two-tier 作为默认质量修复。
  - HP recent 4096 显存复核（2026-06-22）：用 `scripts/measure_vram.sh` 包
    `llama-bench` 直接测 BF16 vs `oscar_int2` 质量候选。8K 结果归档：
    `runs/oscar_int2_hprecent4096_vram_8k_20260622T121415/`，BF16 peak/pp
    `4175 MiB / 3619 tok/s`，oscar_int2 HP4096 `3987 MiB / 1159 tok/s`，
    省约 `188 MiB`。16K 结果归档：
    `runs/oscar_int2_hprecent4096_vram_16k_20260622T121513/`，BF16 peak/pp
    `4859 MiB / 3310 tok/s`，oscar_int2 HP4096 `4117 MiB / 502 tok/s`，
    省约 `742 MiB`。结论：HP4096 质量候选仍保留显存优势，且上下文越长显存优势越
    明显；但速度仍远低于 BF16，后续主瓶颈是 long-context LP bulk prefill。
  - HP4096 + LP two-tier 组合候选（2026-06-22）：在质量候选
    `HP_RECENT=4096` 上叠加远端 LP 降采样 `stride=8,tail=512,weighted=1,mode=end`。
    速度 A/B：HP4096 default pp8192/pp16384 约 `1137/502 tok/s`；叠加
    `s4t512` 约 `1206/643 tok/s`；叠加 `s8t512` 约 `1273/728 tok/s`。对
    `HP4096+s8t512` 跑 10+10 GPQA/GSM8K，结果归档：
    `runs/oscar_int2_hp4096_s8t512_quality_10x10_20260622T122232/summary.md`，
    与 BF16 完全一致：BF16 `2/10,4/10`，组合候选 `2/10,4/10`。显存/速度归档：
    `runs/oscar_int2_hp4096_s8t512_vram_8k16k_20260622T122524/`，8K peak/pp
    `3989 MiB / 1266 tok/s`，16K `4119 MiB / 723 tok/s`；仍低于 BF16 peak，
    但速度仍远低于 BF16。结论：这是当前“质量 smoke 通过且显存优于 BF16”的最快
    候选，但还不能算目标完成；后续要继续优化 long-context LP bulk 或进一步减少
    LP 工作量且扩大质量验证。
  - HP4096 + 更激进 LP two-tier（2026-06-22）：继续扫 `stride=16/32,tail=512`。
    速度：`stride=8` pp8192/pp16384 约 `1277/723 tok/s`；`stride=16` 约
    `1324/873 tok/s`；`stride=32` 约 `1356/969 tok/s`。对
    `HP4096+s32t512` 跑 10+10 GPQA/GSM8K，结果归档：
    `runs/oscar_int2_hp4096_s32t512_quality_10x10_20260622T123106/summary.md`，
    仍与 BF16 完全一致：`2/10,4/10`。显存/速度归档：
    `runs/oscar_int2_hp4096_s32t512_vram_8k16k_20260622T123353/`，8K peak/pp
    `3989 MiB / 1348 tok/s`，16K `4119 MiB / 971 tok/s`。结论：`HP4096+s32t512`
    是当前最快候选，质量 smoke 与 BF16 对齐且显存优于 BF16；但速度仍未超过 BF16
    8K/16K，因此目标仍未完成。后续可以继续扫更大 stride/更小 tail 作为速度上限，
    但必须扩大质量验证。
  - HP4096 + LP two-tier 速度边界继续扫（2026-06-22）：扫
    `stride=32/64/128` × `tail=512/256`。`tail=512` 下 pp8192/pp16384：
    s32 `1329/985`、s64 `1393/1035`、s128 `1253/1016` tok/s；`tail=256` 下：
    s32 `1443/1059`、s64 `1471/1128`、s128 `1338/1093` tok/s。对最快的
    `HP4096+s64t256` 跑 10+10 GPQA/GSM8K，结果归档：
    `runs/oscar_int2_hp4096_s64t256_quality_10x10_20260622T124154/summary.md`，
    仍与 BF16 完全一致：`2/10,4/10`。显存/速度归档：
    `runs/oscar_int2_hp4096_s64t256_vram_8k16k_20260622T124442/`，8K peak/pp
    `3989 MiB / 1448 tok/s`，16K `4119 MiB / 1072 tok/s`。结论：
    `HP4096+s64t256` 是当前最快候选，质量小样本对齐且显存优于 BF16；但速度仍
    未超过 BF16，继续增大 stride 已出现收益递减/回退，后续需扩大质量验证或回到
    LP kernel/格式优化。
  - HP4096 + LP two-tier 更激进 tail 边界（2026-06-22）：继续扫
    `stride=32/64/128` × `tail=128/64`，均为 `weighted=1,mode=end`。`tail=128`
    下 pp8192/pp16384：s32 `1482/1106`、s64 `1500/1115`、s128 `1486/1204`
    tok/s；`tail=64` 下：s32 `1589/1145`、s64 `1512/1208`、s128
    `1518/1268` tok/s。对速度/长上下文更均衡的 `HP4096+s128t64` 跑
    2+2 smoke，输出 `4`，无 `2. 2. 2...` 退化；10+10 GPQA/GSM8K 质量
    smoke 归档：
    `runs/oscar_int2_hp4096_s128t64_quality_10x10_20260622T125254/summary.md`，
    与 BF16 完全一致：BF16 `2/10,4/10`，oscar_int2 `2/10,4/10`。显存/速度
    归档：`runs/oscar_int2_hp4096_s128t64_vram_8k16k_20260622T125545/`，
    8K peak/pp `3989 MiB / 1530 tok/s`，16K `4119 MiB / 1207 tok/s`。
    结论：`HP4096+s128t64` 是当前质量 smoke 对齐且显存优于 BF16 的最快均衡
    候选，但 8K 仍只有 BF16 约 42%，速度目标未完成。
  - HP recent 速度/显存上限探测（2026-06-22）：在 `s128t64` 上继续测
    `HP_RECENT=6144/8192`。8K 单次速度：hp6144 约 `1669 tok/s`，hp8192 约
    `2397 tok/s`；16K 单次速度分别约 `1070/1054 tok/s`，没有继续提升。8K
    显存归档：
    `runs/oscar_int2_hp_sweep_vram_8k_20260622T125850/`，hp6144 peak/pp
    `4149 MiB / 1665 tok/s`，仍略低于 BF16 8K peak `4175 MiB`；hp8192
    peak/pp `4307 MiB / 2311 tok/s`，已经高于 BF16 peak。结论：增大 HP
    recent 可以用显存换 8K 速度，但在“显存也优于 BF16”的约束下，hp6144
    接近上限，hp8192 不满足显存目标；且 16K 没有收益，不能作为默认完成方案。
  - HP4096 + two-tier 过激 tail 负结果（2026-06-22）：继续测
    `tail=32/0` 与 `stride=128/256`。`tail=32` 下 pp8192/pp16384：
    s128 `1531/1236`、s256 `1506/1295` tok/s；`tail=0` 下：s128
    `1380/1229`、s256 `1514/1290` tok/s。结论：更小 tail 没有带来
    8K 明显收益，`s256t32` 只改善 16K 但质量风险更高；`tail=0` 还会回退。
    当前均衡候选仍是 `HP4096+s128t64`；如果只看 8K 速度/显存边界，`HP6144+s128t64`
    可作为实验候选，但已接近 BF16 peak。
  - LP three-tier 摘要候选（2026-06-22）：利用已有 env 三段采样，不改默认。
    配置为 `HP_RECENT=4096`、中段 `stride=64/tail=64`、远端
    `LLAMA_KV_MIXED_VEC_LP_THREE_TIER_MID_TOKENS=2048/4096` 与
    `FAR_STRIDE=128/256/512`。`mid=2048,far=512` 单次速度最好一档：
    pp8192/pp16384 约 `1545/1301 tok/s`；`mid=4096` 整体回退。对
    `mid=2048,far=512` 跑 2+2 smoke，可输出 `4`；10+10 GPQA/GSM8K
    质量 smoke 归档：
    `runs/oscar_int2_hp4096_three_tier_mid2048_far512_quality_10x10_20260622T131002/summary.md`，
    与 BF16 对齐：`2/10,4/10`。显存/速度正式归档：
    `runs/oscar_int2_hp4096_three_tier_mid2048_far512_vram_8k16k_20260622T131242/`，
    8K peak/pp `3989 MiB / 1519 tok/s`，16K `4119 MiB / 1223 tok/s`。
    结论：three-tier 是比单一 stride 更合理的 long-context 摘要方向，质量小样本可行，
    但正式归档收益很小，不足以接近 BF16 速度；不能作为目标完成。
  - HP6144 + three-tier 显存边界复核（2026-06-22）：静态确认 OSCAR2
    `V_rows_per_thread=8` 已经命中 `dequantize_V_oscar2<half,8>` 快路径，不存在简单
    “漏走专用 decode 分支”的修复；不要重复手写 V decode 展开。随后组合
    `HP_RECENT=6144` 与 three-tier。`s128t64` pp8192/pp16384 约 `1677/1115`
    tok/s；three-tier `mid=1024/2048,far=256/512/1024,stride=64,tail=64`
    中，8K 最高单次约 `1699 tok/s`（`mid=2048,far=512/1024`），但 16K 仍只有
    `~1079-1127 tok/s`，明显低于 HP4096 长上下文候选。正式 8K 显存归档：
    `runs/oscar_int2_hp6144_three_tier_mid2048_far512_vram_8k_20260622T132117/`，
    peak/pp `4149 MiB / 1663 tok/s`，仍低于 BF16 8K peak `4175 MiB`，但速度
    仍不到 BF16 一半。结论：HP6144 是 8K 显存边界附近的速度候选，但不适合
    16K/32K；继续调 HP recent/three-tier 参数不能完成目标。
  - HP-only fast-path 诊断（2026-06-22）：临时实现 env-gated
    `LLAMA_KV_MIXED_VEC_HP_ONLY_FAST=1`，当 `ne11_lp_eff == 0` 时绕过 raw mixed
    LP/combine，直接调用现有 F16/F16 `flash_attn_ext_vec`，用于测试 HP 全覆盖时
    mixed 框架开销。HP8192 8K：默认 mixed `~2328 tok/s`，fast path `~2359 tok/s`，
    仅小幅提升；16K fast path `~1015 tok/s`。该路径已撤回，构建恢复通过。
    结论：HP-only 情况仍远低于 BF16 8K `~3619 tok/s`，主要差距不只是 mixed
    combine kernel，而与 HP cache/mask/graph 形态或整条 mixed graph 的数据布局相关；
    不要保留 `HP_ONLY_FAST` 或继续在该短路上小修。
  - HP allocation padding 正收益（2026-06-22）：发现 HP cache 总长度
    `n_kv_sink + n_kv_recent` 可能不是 `FATTN_KQ_STRIDE=256` 对齐，例如
    `64+8192=8256`，导致 HP-only F16 attention 不能稳定命中 WMMA/F16 快路径。
    新增 env-gated `LLAMA_KV_HP_ALLOC_PAD=256`，在 KV cache 构造时把 HP buffer
    总长度 pad 到 256 倍数，额外槽保持空，由 HP mask 置 `-inf`。构建通过。
    调试确认 HP8192 时 `n_hp=8448`、最终 `hp_ne2=8448`。速度：
    HP8192 8K 从默认约 `2614 tok/s` 提到 `~3081-3227 tok/s`，但显存会超过
    BF16，不作为目标候选。显存内候选：
    `HP6144+s128t64+ALLOC_PAD=256` 8K peak/pp
    `4161 MiB / 1755 tok/s`，仍略低于 BF16 8K peak `4175 MiB`；
    16K peak/pp `4291 MiB / 1105 tok/s`，低于 BF16 16K peak `4859 MiB`
    但速度不如 HP4096 长上下文候选。`HP4096+s128t64+ALLOC_PAD=256`：
    8K `4001 MiB / 1605 tok/s`，16K `4131 MiB / 1189 tok/s`。质量 smoke：
    `runs/oscar_int2_hp6144_pad256_s128t64_quality_10x10_20260622T133826/summary.md`，
    与 BF16 对齐：GPQA/GSM8K `2/10,4/10`。结论：`LLAMA_KV_HP_ALLOC_PAD=256`
    是应保留的 env-gated 正收益，解决了 HP F16 路径未对齐导致的速度损失；
    但即使最佳显存内 8K 只有 `~1755 tok/s`，目标仍未完成。
  - F16/BF16 cache 对照（2026-06-22）：同一 Granite 1B、8K/16K prefill 下，
    普通 `f16/f16` KV cache 不慢于 `bf16/bf16`，近期单次约为 f16 8K/16K
    `4043/3549 tok/s`，bf16 8K/16K `3857/3214 tok/s`。结论：OSCAR INT2
    的 HP 段使用 F16 本身不是速度根因；差距更可能来自 HP cache/mask/layout
    或 mixed graph 触发的 FA kernel 选择与数据布局。
  - HP near-full + allocation padding 显存边界（2026-06-22）：继续测
    `HP_RECENT=7168/7680/8064`、`LLAMA_KV_HP_ALLOC_PAD=256`、`s128t64`。
    8K 显存归档 `runs/oscar_int2_hp_near_full_pad_vram_8k_20260622T134502/`：
    hp7168 peak/pp `4241 MiB / 2089 tok/s`，hp7680 `4281 MiB / 2355 tok/s`，
    hp8064 `4301 MiB / 1946 tok/s`。这些都高于 BF16 8K peak `4175 MiB`，
    因此不满足“显存优于 BF16”；在显存约束内，`HP6144+ALLOC_PAD=256`
    基本是当前 HP recent 上限。下一步必须减少 LP/mixed 开销或修正 HP/mask
    的 FA dispatch/layout，而不是继续加 HP window。
  - FA kernel-selection 调试（2026-06-22）：新增 env-gated
    `LLAMA_CUDA_FATTN_DEBUG=1`，只打印 CUDA FA kernel 选择，不改变默认路径。
    8K prompt 下，普通 BF16/BF16 cache 的 prefill 走 `kernel=mma_f16`；
    `HP_RECENT=8192 + LLAMA_KV_HP_ALLOC_PAD=256` 的 HP F16 view 也能走
    `kernel=mma_f16`（例如 HP view `K/V type=f16 ne[1]=8448`，`gqa_opt=1`）。
    结论：HP alloc padding 已修正 HP view 对齐/dispatch；HP F16 attention 本身
    可以命中快路径。当前剩余速度差主要来自 OSCAR2 bulk 的写入/量化与 LP/mixed
    维护成本，而不是 HP 段误走慢 FA kernel。
  - 失败诊断勿重复（2026-06-22）：尝试临时 `LLAMA_KV_HP_SKIP_LP_STORE_DIAG`
    跳过 LP OSCAR2 cache 写入以估算 set_rows/quantize 上限。直接跳过会使
    `self_k_idxs` graph input 无 buffer；尝试用零依赖挂住 input 又触发 CPU `dup`
    fatal。该诊断代码已撤回，构建恢复通过。不要再用“跳过 LP store”这种破坏
    graph/input 分配的方式测上限；如需定位 LP 写入成本，应改为正式计时/可执行的
    set_rows microbench 或优化 OSCAR2 set_rows kernel 本身。
  - OSCAR2 set_rows parallel v1（2026-06-22）：新增 env-gated
    `LLAMA_KV_OSCAR2_SET_ROWS_PAR=1|k|v`，只影响 `GGML_TYPE_OSCAR2_KV`
    的 CUDA `set_rows`，默认关闭。v1 用一个 128-thread block 量化一个
    `block_oscar2_kv`，并行算 mean/sigma、按 K/V centroid pack `qs/rs`，fallback
    仍是旧 generic “一个线程串行量化 128 维”的路径。构建通过，`llama-completion`
    smoke 正常输出 `2+2=4`，无 `2. 2. 2...` 退化。A/B：
    `HP6144+s128t64+ALLOC_PAD=256` 下 p512 从约 `2175-2181` 提到
    `2317-3339 tok/s`（短上下文固定写入成本明显）；p2048 单次有正有负
    (`2403→2534`，另一次 `3407→2555`)；p8192 拆分诊断中 K-only/V-only
    单次可到 `~1801/~1790`，但 all 反而 `~1535`，同批 off 波动范围
    `1702/1774/1822` 覆盖收益。结论：OSCAR2 KV store/quantize 确实是短上下文
    成本来源，但当前 parallel v1 对 8K 不稳定且不足以靠近 BF16，不能默认；
    后续若继续写入方向，应做更正式的 set_rows microbench 或更低开销的 K/V
    专用 packing，不要把 `LLAMA_KV_OSCAR2_SET_ROWS_PAR=1` 作为最终方案。
  - LP sampling 与 LP store 速度上限诊断（2026-06-22）：在显存内候选
    `HP_RECENT=6144 + HP_ALLOC_PAD=256` 上继续把 LP two-tier 采样推到
    `stride=256/512/1024`、`tail=64/32/0`。8K 最好仍只有约 `1748 tok/s`，
    `stride=1024/tail=32` 同批约 `1735 tok/s`，远低于 BF16 同批
    `3817 tok/s`。结论：LP 已经被采到很稀，继续减少 LP token 数收益到顶，
    不能完成速度目标。
  - HP8192/跳过 LP store 速度上限诊断（2026-06-22）：新增 env-gated
    `LLAMA_KV_HP_SKIP_LP_STORE_DIAG=1`，只有当 HP KV 已覆盖当前 KV used 时才不创建
    LP `self_k_idxs/self_v_idxs`、跳过 LP OSCAR2 set_rows，用于估算“完全省掉 LP 写入”
    的速度上限；不作为正确最终方案，因为 recent token 未来滑出 HP 后仍需要 LP bulk。
    HP8192 + alloc pad + 极稀 LP `s1024/t32` 下，8K skip0/skip1 约
    `2744/2808 tok/s`，只提升约 2%，仍低于 BF16 `3817 tok/s`，且 HP8192 peak
    已超过 BF16 显存目标。结论：LP store 不是当前 8K 主瓶颈；即便 HP 全覆盖且
    省掉 LP 写入，整条 oscar_int2/HP graph 仍明显慢于 BF16。后续不要继续在
    HP window、LP sampling 或简单 LP store skip 上投入，除非改成真正的
    cold-LP/HP-eviction 延迟量化语义。
  - OSCAR INT2 收敛计划（2026-06-22）：当前目标从“继续堆 kernel/env 实验”
    收敛为“判断纯 INT2 bulk 是否值得继续”。冻结审计目录：
    `runs/oscar_int2_convergence_20260622/`，包含 dirty 状态、关键 diff 与
    env 引用。后续测试矩阵只保留：
    `baseline_bf16`、`oscar_int2` 显存内 HP6144、`oscar_int2` near-full HP
    速度上限、`oscar_int4`。停止重复 HP window、mask、CUDA graph、LP sampling、
    set_rows parallel v1、旧 dedicated LP kernel、split/graph split、V helper
    微调。判定规则：如果显存内 `oscar_int2` 仍只能约 `1.7-1.9k tok/s`，
    而 BF16 8K 约 `3.6-4.0k tok/s`，则不再把“纯 scalar INT2 bulk + HP window”
    当最终路线；下一步只做格式级候选，优先 K-side residual/two-tier，若质量/速度
    仍不够，再转 K3/V2 作为非纯 INT2 备选。`LLAMA_KV_HP_VIEW_TIGHT=1` 是
    新增 env-gated 诊断/候选：HP8192 8K 单次可从约 `2641` 提到 `3461 tok/s`，
    说明 HP-only view 裁剪有效；但 HP8192 显存仍不满足 BF16 peak 目标，
    所以不能作为最终解。
  - K-Affine Sidecar 最后一轮格式候选（2026-06-23）：新增离线 probe
    `scripts/probe_oscar2_k_affine4.py`，使用 Granite Q/K dump 对每个 128-d
    K block 分 4 组拟合 `K_g ~= a_g*Khat_g + b_g`。Gate 1 通过：
    `runs/oscar2_k_affine4_probe_current/summary.md` 中 rotated KQ score
    NMSE 从约 `0.0747` 降到 `0.0540`，平均改善约 `33%`；plain 平均改善约
    `59%`。随后做过 CUDA/KV 原型：为 OSCAR2 K 分配 F16 sidecar、set_rows
    写 `a0..a3,b0..b3`、raw mixed FA KQ 读取 sidecar。小 bench 能 build/run，
    但 Gate 2 失败：同场 8K BF16 pp 约 `3494 tok/s`，`oscar_int2` affine4
    off/on 约 `385/375 tok/s`，远低于 BF16 80% 门槛且 sidecar 略慢。按计划
    运行时接入已撤回，源码中 `K_AFFINE4/k_affine4/mixed_with_k_sidecar`
    只剩离线 probe 引用；不要继续把 affine4 sidecar 接回默认 CUDA hot path。
    结论：affine4 能降低离线 KQ 误差，但不能解决当前 llama.cpp mixed INT2
    速度瓶颈；纯 `oscar_int2` 路线冻结，交付线应回到 `oscar_int4` 或另起
    非纯 INT2 的 K3/V2 可行性分支。
  - Delayed LP materialization 最后一轮结构候选（2026-06-23）：按最终计划实现过
    `LLAMA_KV_OSCAR2_DELAYED_LP=1` 原型，只在 CUDA/Granite D=128/OSCAR2 KV/HP
    prefill attention 下启用。实现方式是从 `slot_info.hp_batch_idxs` 生成 cold-only
    LP 写入行，只把非 HP rows 通过 `ggml_get_rows + ggml_set_rows` 写入 OSCAR2 LP
    cache，HP rows 只写 BF16 HP cache。初版因把 2D cold rows 直接传入 `cpy_k/cpy_v`
    触发 `ggml_set_rows` 宽度断言，修成 reshape 回 `[head_dim,n_head,n_cold]` 后能跑。
    结果：p512 delayed-LP `1355 tok/s`，p2048 `1311 tok/s`，p8192 `588 tok/s`；
    同场 BF16 p8192 `3529 tok/s`，80% gate 为 `2823 tok/s`。Gate B 明确失败，
    因此未跑质量/16K/32K。运行时 hot path 已撤回，源码中不应再有
    `LLAMA_KV_OSCAR2_DELAYED_LP`/`lp_cold` 残留。最终记录：
    `docs/OSCAR_INT2_FINAL_CONVERGENCE_PLAN.md`。结论：纯 `oscar_int2` 暂时冻结，
    不再重复 delayed LP、HP skip LP store、K-affine sidecar、HP window/mask/CUDA
    graph/V helper/LP sampling/q2 scalar/旧 tile kernel 小修；后续交付以
    `oscar_int4` 为主。

## OSCAR INT4 delivery refresh（2026-06-24）

- 本轮只复测 `baseline_bf16` 与 `oscar_int4`，不重新打开 INT2/INT3。构建通过：
  `cmake --build third_party/OSCAR/build-cuda -j 4 --target llama-bench llama-server llama-cli`。
- 质量：server eval 50+50 复现了弱 harness 问题，BF16 自己也接近全 0，因此不作为
  最终质量信号。改用历史可靠的 CLI harness 复测：
  `runs/oscar_int4_bf16_cli_quality_50_20260624/`。结果 BF16 GPQA/GSM8K
  `11/50,16/50`，`oscar_int4` 同样 `11/50,16/50`。结论：INT4 与 BF16 同档，
  没有 INT2/INT3 的重复/空输出退化。
- 8K decode-heavy bench：`runs/oscar_int4_bf16_8k_n256_decode_20260624/`。
  BF16 `pp/tg=3283.0/61.9 tok/s`、peak `5508 MiB`、KV pool `640 MiB`；
  `oscar_int4` `pp/tg=2891.0/57.8 tok/s`、peak `5904 MiB`、KV pool `180 MiB`。
  该 8K n256 run 的 INT4 peak 受生成长度/arena 行为影响高于 BF16，不作为显存交付口径。
- 32K decode-heavy bench：`runs/oscar_int4_bf16_32k_n64_decode_20260624/`。
  BF16 `pp/tg=2319.0/63.3 tok/s`、peak `6143 MiB`、KV pool `2560 MiB`；
  `oscar_int4` `pp/tg=2344.8/59.7 tok/s`、peak `4307 MiB`、KV pool `720 MiB`。
  32K peak 下降约 `1836 MiB`，基本等于 KV pool 下降 `1840 MiB`；prefill 持平/略高，
  decode 约为 BF16 的 `94%`，已经超过 80% gate。
- CUDA graph A/B：`runs/oscar_int4_32k_n64_graph_off_20260624/` vs
  `runs/oscar_int4_32k_n64_graph_on_opt_20260624/`。单 case `oscar_int4`
  32K n64 下 graph off `pp/tg=2395.2/48.1`，graph on+opt `2392.6/46.9`，
  无收益且低于同场默认双 case tg。不要基于这轮 A/B 修改 INT4 默认 graph 策略。
- 当前建议：`oscar_int4` 已达交付状态；不要为 INT4 decode 做高风险 CUDA kernel 小修。
  若需要继续验证，优先把 CLI 质量扩到 100+100；只有发现 32K tg 稳定低于 BF16 80%
  时再考虑 q4 decode hot path 优化。
