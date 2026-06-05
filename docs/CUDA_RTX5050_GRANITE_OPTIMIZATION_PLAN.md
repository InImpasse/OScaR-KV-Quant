# CUDA 优化计划：RTX 5050 Laptop（8GB）+ Granite 4.0 1B BF16

本文档面向本仓库的 **llama.cpp fork**（`third_party/OSCAR/`），在 **NVIDIA RTX 5050 Laptop、约 8GB 显存、compute capability 12.0（Blackwell）** 上，用 **`checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`** 做基准与调参。

---

## 1. 目标与范围

| 层级 | 目标 |
|------|------|
| **短期（本机可完成）** | CUDA 原生编译、稳定跑通 `llama-cli` / `llama-bench`；在显存约束下扫 **KV 类型**（`f16` / `q8_0` / `q4_0` / `q2_0`）的 **吞吐与峰值显存**；建立可复现实验目录与配置表。 |
| **中期** | 固定 workload（`CONTEXT`、prompt 长度、`n_gpu_layers`、`-fa`）下找 **tok/s 与 OOM 边界**；必要时用 `nsys` 做一次热点确认（可选）。 |
| **长期（若要做 OSCAR INT2「校准精度」）** | 需 **Qwen3 等带 baked rotation 的 `*-rot-kv.gguf`**，且 CUDA 上 **全量化 V 的 `q2_0` + Flash Attention** 仍可能受 fork 内核完整性约束——与 Granite「格式压测」分开立项（见第 6 节）。 |

**Granite 说明**：当前 BF16 GGUF **未带 OSCAR 校准旋转**；`q2_0` 在此模型上主要验证 **KV 存储路径与性能**，**不**代表论文级 OSCAR INT2 精度结论。

---

## 2. 环境基线（已测）

- **GPU**：NVIDIA GeForce RTX 5050 Laptop GPU，约 **8151 MiB**，**compute capability 12.0**。
- **模型路径（建议）**：`checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`（相对仓库根目录）。

---

## 3. Phase 0 — 构建与可运行性（优先）

### 3.1 CUDA 构建

在仓库根目录执行（按本机 CUDA Toolkit / 驱动调整）：

```bash
cd /home/lenovo/project/OSCAR-KV-Quant

LLAMACPP_CMAKE_ARGS="-DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
  ./scripts/build_llamacpp.sh
```

说明：

- **`native`**：让 CMake 针对当前 GPU 生成 SASS，避免默认架构列表遗漏 **sm_120** 导致意外走 PTX JIT 或缺核。
- 若 `native` 在个别工具链上报错，可改为显式 **`120`** 或与 fork 文档一致的 `CMAKE_CUDA_ARCHITECTURES` 写法后再编一次。

### 3.2 单次冒烟（推理）

```bash
MODEL=/home/lenovo/project/OSCAR-KV-Quant/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
CONTEXT=8192 KV_TYPE=f16 PREDICT=64 N_GPU_LAYERS=999 \
PROMPT="Say hello in one sentence." \
./scripts/run_llamacpp.sh
```

成功后把 **`KV_TYPE`** 依次改为 `q8_0`、`q4_0`、`q2_0`；若 **`q2_0` 在首包或构图阶段崩溃**，记录日志（可能与 **Flash Attention + `Q2_0` 在 CUDA 上的支持矩阵** 有关，见第 6 节）。

---

## 4. Phase 1 — 8GB 显存下的 KV 基准扫参

### 4.1 使用现有 `bench_kv_cache.sh`

脚本默认 **`CONTEXT=32768`**、**`PROMPT_TOKENS=4096`**。在 8GB 上若 OOM，**每次只改一个变量**向下探：

1. **`CONTEXT`**：`32768` → `16384` → `8192`
2. **`PROMPT_TOKENS`**：`4096` → `2048` → `1024`
3. **`N_GPU_LAYERS`**：`999`（全层 GPU）→ 减少 offload 层数，把部分算子留在 CPU（更慢但省显存）

示例（从保守配置开始）：

```bash
MODEL=/home/lenovo/project/OSCAR-KV-Quant/checkpoints/gguf/granite-4.0-1b-base-bf16.gguf \
CONTEXT=16384 PROMPT_TOKENS=2048 GEN_TOKENS=512 N_GPU_LAYERS=999 \
KV_TYPES=f16,q8_0,q4_0,q2_0 \
./scripts/bench_kv_cache.sh
```

输出在 `runs/llamacpp_kv_<UTC时间>/`，含各 `*.json` 与 `config.txt`。

### 4.2 建议记录的指标

- **`pp` / `tg`**（prefill vs decode）与 **总 tok/s**（以 `llama-bench` JSON 为准）。
- 每次 run 旁注：**峰值显存**（`nvidia-smi` 另开终端观察即可）。
- **Flash Attention**：若 CLI 支持，显式对比 `-fa on` / `-fa off`（注意：**量化 V 时 fork 可能强制依赖 FA**，以实际报错为准）。

### 4.3 期望趋势（用于 sanity check）

在相同 `CONTEXT` 与 batch 设定下，通常 **KV 从 `f16` → `q8_0` → `q4_0` → `q2_0`** 会 **降低 KV 占用**；吞吐变化取决于 **是否仍受内存带宽或 FA 内核路径主导**，不一定单调变快。

---

## 5. Phase 2 — 参数化「优化」搜索空间（本机）

在 **不 OOM** 的前提下，对下列维度做小网格搜索（每次只动 1–2 个维度，避免组合爆炸）：

| 变量 | 建议扫法 |
|------|-----------|
| **`CONTEXT`** | 在业务目标上下文（如 8k / 16k）附近取 2–3 个点。 |
| **`PROMPT_TOKENS` / `GEN_TOKENS`** | 固定 prefill-heavy 或 decode-heavy 场景各一组。 |
| **`N_GPU_LAYERS`** | 全 GPU vs 减层（缓解碎片/峰值显存尖峰）。 |
| **KV 类型** | 以 `f16` 为 baseline，对比 `q8_0` / `q4_0` / `q2_0` 的 **显存峰值 + tok/s**。 |

**优化判据（可二选一或加权）**：

1. **固定 CONTEXT 与质量接受范围** → 最大化 **decode tok/s**；或  
2. **固定最低 tok/s** → 最大化 **CONTEXT**（在 8GB 内）。

---

## 6. Phase 3（并行/后续）— OSCAR INT2 全链路与 CUDA

若目标从「Granite 上压 KV 格式」升级为 **「OSCAR 校准 INT2 + CUDA」**：

1. **模型**：使用文档中的 **`qwen3-4b-thinking-*-rot-kv.gguf`**（或自行 bake 的 rot-kv），并配置 `LLAMA_KV_*` 环境变量（见 `third_party/OSCAR/README.md`）。
2. **正确性**：以 **CPU（`-ngl 0`）** 或文档推荐组合为参考，对 CUDA 输出做 **对照**（同 prompt、同 seed 若可）。
3. **内核**：本 fork 的 CUDA **Flash Attention** 对 **`GGML_TYPE_Q2_0`** 的支持需与 `fattn.cu` / vec-tile-mma 路径核对；若尚未覆盖，需要 **开发补核** 后才能认为 **「K+V 全 `q2_0` + `-fa on`」** 在 CUDA 上生产可用。

该项与 Granite 8GB 基准 **解耦**：Granite 计划可先闭环第 3–5 节。

---

## 7. 交付清单（你可以用来勾选）

- [ ] `LLAMACPP_CMAKE_ARGS` 含 **`GGML_CUDA=ON`** 且架构覆盖 **sm_120**，`llama-cli` / `llama-bench` 可执行。  
- [ ] `MODEL` 指向 **`checkpoints/gguf/granite-4.0-1b-base-bf16.gguf`**，`f16` 冒烟通过。  
- [ ] 在 **不 OOM** 的最大 `(CONTEXT, PROMPT_TOKENS, N_GPU_LAYERS)` 下完成 **`KV_TYPES=f16,q8_0,q4_0,q2_0`** 一轮 bench，结果落盘 `runs/`。  
- [ ] 文档化一条 **推荐默认配置**（例如 16k ctx + 2k prefill + 全层 GPU）供日常回归。  
- [ ] （可选）`nsys profile` 一次，确认瓶颈在 FA / GEMM / 还是 host 侧。  

---

## 8. 参考命令速查

| 动作 | 命令 |
|------|------|
| 编译 | `LLAMACPP_CMAKE_ARGS="-DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" ./scripts/build_llamacpp.sh` |
| 单次推理 | `MODEL=.../granite-4.0-1b-base-bf16.gguf CONTEXT=8192 KV_TYPE=q4_0 ./scripts/run_llamacpp.sh` |
| KV 对比 | `MODEL=... CONTEXT=... PROMPT_TOKENS=... ./scripts/bench_kv_cache.sh` |

---

*文档版本：与仓库 `scripts/*.sh` 行为一致时可长期使用；若上游 fork 更新 Flash Attention / `q2_0` 矩阵，请复查第 6 节。*
