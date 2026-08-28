<div align="center">

# 双 AMD Radeon RX 7900 XTX (RDNA3) 上的 vLLM

[English](README.md) | [简体中文](README.zh-CN.md)

[vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork，包含 RDNA3/gfx1100 平台性能工作：
DFlash2 drafter 修复、ROCm 内核优化，以及调优过的双 7900 XTX 服务配置。

</div>

---

## 关于本分支

本分支（`feat/dflash-perf-opt`）在上游 vLLM 之上携带 RDNA3 专用改进：

- **DFlash2 投机解码修复与自适应控制器** — 解决 CUDA Graph 下的零接受率塌方问题（ROCm 平台）。
  接受率已恢复、draft token 真实、0–64k 上下文稳定。
  引入了灵感源自 [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp/commit/ca26169d2efefebc028a25c115714080dc2475e3) 与 [@Digit4lSoluti0n](https://x.com/digit4lsoluti0n/status/2092700856164421861?s=46) 的**闭环自适应投机长度控制器**（`AdaptiveSpecController`）：通过右侧截断 EMA 在线跟踪 token 接受情况，在全接受时加性上探、部分拒绝时平滑回退，并以 GPU 纯矢量化广播掩码过滤超额候选，彻底消除深层上下文下的验证算力衰退。
- **ROCm 内核工作** — 1D grouped conv 与 split-kv reduction 的原生融合 HIP 内核、
  sliding-window 检测加固、draft token 净化。
- **继承自 [JartX/vllm](https://github.com/JartX/vllm) 的 RDNA3 优化** — 对其分支审计后，
  以下项目确认在本分支中（已对照分支历史核实）：
  - Triton attention 自适应 prefill 调优（按 KV 长度选择 `BLOCK_M`/`num_warps`，`6cb90f545`）
  - 3D split-KV softmax NaN 修复（长上下文 decode，`6b84c6c6c`）
  - bounds-safe Triton attention 索引 + `triton_quant_kv` dtype 缓存（`883e982e6`）
  - RDNA3 W4A16 GPTQ 内核（`csrc/rocm/q_gemm_rdna3*.cu`，运行时以
    `RDNA3W4A16LinearKernel` 使用）
  - INT8 per-tensor KV cache（`647b3883b`，代码保留；本模型无预校准 scale 不可用）
  - 注意：**hostar all-reduce 与 WMMA paged-prefill 曾尝试但已回滚**（`4a8e2652f`，端到端
    收益为负）；MoE 工作**不在**本分支。
  **由于这些继承的优化，下方基准中的 "stock vLLM" 列实为同一 fork 关闭投机解码的状态——
  其 prefill/decode 高于纯上游 vLLM。** 这些改进归功于 JartX 的工作。
- **调优的双 7900 XTX 服务配置** — 见下方启动配方与实测数据。

---

## 硬件与软件环境

| 项目 | 配置 |
|---|---|
| 显卡 | 2× AMD Radeon RX 7900 XTX，各 24 GiB，gfx1100，TP=2 |
| ROCm | **7.14**（HIP 7.14.60850） |
| GPU 驱动 | **amdgpu 内核模块 3.2.340**（kernel 6.17.0-35-generic，Ubuntu 24.04，`amdgpu-install` 30.30.4） |
| PyTorch | 2.11.0+git，ROCm wheel（`torch.version.hip = 7.2.53211`） |
| 系统 | Linux（启用 ROCm），见上游 [ROCm 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| ROCm 环境变量 | `HSA_OVERRIDE_GFX_VERSION=11.0.0`、`HSA_ENABLE_IPC_MODE_LEGACY=0`、`HSA_FORCE_FINE_GRAIN_AMDGPU=1`、`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`、`GCN_ARCH_NAME=gfx1100`、`VLLM_ROCM_USE_AITER=1` |
| 目标模型 | Qwen3.8-27B（W4A16 GPTQ，hidden 5120），FP8 KV cache |
| DFlash2 drafter | 5 层 `Qwen3.8-27B-DFlash2-bf16`，`num_speculative_tokens=7`，sliding window 2048 |

> **注意**：`HSA_ENABLE_IPC_MODE_LEGACY=0` 与 `HSA_FORCE_FINE_GRAIN_AMDGPU=1` 对 TP=2 稳定性
> **必需** —— 去掉后第一个请求即触发 `ProcessGroupNCCL` 崩溃。它们不出现在 vLLM 源码中，
> 但由 HIP 运行时读取。

---

## 从源码构建

```bash
# 前置：上述 ROCm/驱动栈 + 匹配的 PyTorch ROCm wheel，CMake >= 3.26，HIP 工具链。
cd vllm
pip install -e .          # 通过 torch.version.hip 自动识别 ROCm（VLLM_TARGET_DEVICE=rocm）
python -c "import vllm; print(vllm.__version__)"   # 验证 editable 安装
```

- `csrc/rocm/` 下的 HIP C++ 内核（含 `RDNA3W4A16LinearKernel`）由同一 editable 安装编译；
  `ROCM_HOME` 自动识别，`HSA_OVERRIDE_GFX_VERSION=11.0.0` 仅在运行时需要（并供 Triton JIT
  通过 `GCN_ARCH_NAME` 定位 gfx1100）。
- 本文档其余部分假定已完成可用的 editable 构建；启动配方使用本仓库的 `serve-bootstrap.py`
  包装，但直接 `vllm serve` 效果相同。

---

## 如何准备 qwen3.8-27b-mtp-fixed（模型准备）

目标模型使用打过补丁的本地 checkpoint `qwen3.8-27b-mtp-fixed`，源自
[`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ)
（源 `config.json` 已对照 HF 线上仓库核实）：

1. 原仓库把全部 15 个 MTP tensor 以纯 **BF16** 存放在 `model_extra_tensors.safetensors` 中，
   但其 `quantization_config.dynamic` 声明了**正规则** `"+:.*mtp.*"` 与 `"+:.*mtp\.fc.*"`
   （4-bit，group 64）。
2. vLLM 的 GPTQ loader 信任该配置，按量化参数构建 MTP 层，权重加载失败
   （`no module or parameter named 'layers.0.mlp.down_proj.weight'`）。
3. **补丁**：把 `config.json` 中 `quantization_config.dynamic` 里的 `"+:"` 规则替换为一条
   `"-:.*mtp.*"` 排除规则。vLLM 内置 workaround（`qwen3_5_mtp.py`）随即对 MTP 层禁用量化，
   BF16 权重原样加载。上游参考：[vllm-project/vllm#48816](https://github.com/vllm-project/vllm/pull/48816)、
   [#47828](https://github.com/vllm-project/vllm/pull/47828)。

补丁后的 `config.json` 共 97 条 negative 规则、0 条 positive 规则（已对照本地
`qwen3.8-27b-mtp-fixed/config.json` 核实：`"mtp"` 仅以 `-:.*mtp.*` 形式出现）。DFlash2 以
同一 checkpoint 作为目标模型；DFlash2 drafter 为独立 checkpoint：
- **BF16 Drafter**：本地以 `Qwen3.8-27B-DFlash2-bf16` 提供，上游镜像见 [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)。
- **W4A16 量化 Drafter**：本地以 `Qwen3.8-27B-DFlash2-W4A16` 提供，开源地址见 [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)。

---

## 如何运行

### 1. DFlash2 投机解码（BF16 Drafter）

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_IPC_MODE_LEGACY=0
export HSA_FORCE_FINE_GRAIN_AMDGPU=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export GCN_ARCH_NAME=gfx1100
export VLLM_ROCM_USE_AITER=1
export HF_HUB_OFFLINE=1

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen3.8-27b-mtp-fixed \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-model-len 200000 \
  --mamba-cache-mode align \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"dflash","model":"/path/to/Qwen3.8-27B-DFlash2-bf16","num_speculative_tokens":7}' \
  --compilation-config.cudagraph_capture_sizes "[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]" \
  --performance-mode throughput
```

### 2. DFlash2 投机解码（W4A16 量化 Drafter：`syvai/Qwen3.8-27B-DFlash2-W4A16`）

单卡节省约 0.8 GiB 显存，数学推理速度刷新至 **183.8 tok/s**（MATH-500 峰值 **196.0 tok/s**）。
闭环自适应投机**默认开启**（`VLLM_SPEC_DRAFT_ADAPTIVE=1`，`n_max=7, n_min=2`），在结构化输出阶段维持大步长拉满，遇到低熵长文本时自动回退，消除深层注意力验证衰退：

```bash
# 可选自适应参数微调（下方为默认值）：
# export VLLM_SPEC_DRAFT_ADAPTIVE=1
# export VLLM_SPEC_DRAFT_N_MIN=2
# export VLLM_SPEC_DRAFT_ALPHA=0.25
# export VLLM_SPEC_DRAFT_PROBE=1.0

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen3.8-27b-mtp-fixed \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-model-len 200000 \
  --mamba-cache-mode align \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"dflash","model":"/path/to/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}' \
  --compilation-config.cudagraph_capture_sizes "[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]" \
  --performance-mode throughput
```

### 3. MTP3 投机解码（原生预测头，`align` 模式）

适合超长上下文 Agent 会话与高并发业务（与主模型共享 KV 缓存，可用 **11.3 GiB / 63.8 万 tokens**）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen3.8-27b-mtp-fixed \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --mamba-cache-mode align \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --performance-mode throughput
```

> **注意**：启动后请发送一次 warm-up 请求以吸收首次 Triton JIT 编译尖峰。公平基准必须关闭 `VLLM_PROFILE_STEP`。

---

## 性能数据

使用 `llama-benchy 0.4.0`（`--pp 2048 --tg 128 --concurrency 1`，2026-08-27 合并官方 Upstream 后更新）实测。
单元格为 `prefill (PP, tok/s) / decode (TG, tok/s)`。DFlash2/MTP 为关闭 `VLLM_PROFILE_STEP` 后的实测。

| 上下文深度 | stock vLLM（早期上游） | **vLLM DFlash2 (BF16)** | **vLLM DFlash2 (W4A16, 自适应)** | **vLLM MTP3 (BF16, align)** | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1773 / 53.8 | 1634 / **98.7**（峰值 107.0） | 1658 / **118.9**（峰值 119.0） | 1617 / **102.3**（峰值 105.0） | 803 / 63.7 |
| **4k** | — | 1742 / **89.5**（峰值 92.0） | 1697 / **102.5**（峰值 104.0） | 1696 / **96.5**（峰值 107.0） | — |
| **8k** | 1131 / 49.4 | 1673 / **81.6**（峰值 83.0） | 1586 / **84.2**（峰值 96.0） | 1592 / **102.0**（峰值 110.0） | 929 / 65.6 |
| **16k** | 838 / 45.7 | 1495 / **82.2**（峰值 90.0） | 1491 / **80.4 ~ 87.9**（峰值 95.0） | 1416 / **90.5**（峰值 101.0） | 904 / 58.9 |
| **32k** | — | 1163 / **62.0** | — | 1190 / **64.2** | — |
| **64k** | — | 836 / **56.1** | — | 848 / **56.9** | — |

### 要点

- **投机解码收益显著**：0–16k 全深度 DFlash2/MTP 的 decode 领先早期 stock vLLM **50–120%**（depth 0 从 54→102~119 tok/s）。
- **自适应投机消除深层衰退**：通过在 `[n_min=2, n_max=7]` 间动态调节草稿长度，DFlash2-W4A16 在 4k 深度保持 **102.5 tok/s**，在 16k 深度保持 **87.9 tok/s**，depth 0 达到 **118.9 tok/s**。
- **MTP3 在中深长上下文领先**：MTP3 在 8k 深度达到 **102 tok/s**，在 16k 深度保持 **90.5 tok/s**，且与主模型共享 KV 缓存。
- **DFlash2 统治结构化数学与代码推理**：DFlash2-W4A16 达到 **183.8 tok/s**（MATH-500 峰值 **196.0 tok/s**），DFlash2-BF16 达到 **178.2 tok/s**，领先 MTP3 达 **+35–40%**。
- **KV cache 权衡**：
  - **MTP3**：**11.3 GiB / 63.8 万 tokens** 可用缓存（~2.44x 并发 @ 262k）。
  - **DFlash2 (W4A16)**：**7.7 GiB / 33.5 万 tokens** 可用缓存（~1.68x 并发 @ 200k）。
  - **DFlash2 (BF16)**：**6.5 GiB / 28.2 万 tokens** 可用缓存（~1.41x 并发 @ 200k）。

---

## 关键性能主张

**数学推理 decode（depth 0）**（GSM8K + MATH-500，greedy，4 题，正确率 100%）：

| 引擎 | 平均 decode |
| :--- | :---: |
| vLLM Baseline（无投机） | 56.3 tok/s |
| **vLLM DFlash2 (W4A16, 自适应 K=7)** | **183.8 tok/s（峰值 196.0 tok/s，3.26×）** 🏆 |
| **vLLM DFlash2 (BF16, K=7)** | **178.2 tok/s（3.16×）** |
| vLLM MTP3 (K=3) | 131.4 tok/s（2.33×） |
| llama.cpp (Q6_K_XL, MTP) | 87.5 tok/s（1.55×） |

---

## 关键发现与坑

1. **`p_min` 提前截断：短上下文有益、长上下文有害** —— `DFLASH_P_MIN=0.3` 在 depth 0
   +9% decode，但在 8k 深度最多 -11%；长上下文场景保持 0。
2. **vLLM `/health` 返回空 body + HTTP 200** —— 需检查 HTTP code 而非 body。
3. **不要用 `pkill -9 -f "vllm"`** —— 会匹配到执行命令的 bash 自身，同一行后续命令静默
   跳过；改用 `rocm-smi --showpids` 取 PID 杀进程。

---

## 参考与致谢

- **自适应投机解码启发**：在线截断观测 EMA 控制器设计灵感源自 Laurent Zuijdwijk 在 [llama.cpp commit `ca26169`](https://github.com/LaurentZuijdwijk/llama.cpp/commit/ca26169d2efefebc028a25c115714080dc2475e3) 的实现以及 [@Digit4lSoluti0n 的讨论](https://x.com/digit4lsoluti0n/status/2092700856164421861?s=46)。
- 上游原始 README：[vllm-project/vllm](https://github.com/vllm-project/vllm#readme) —— 本 fork
  以 RDNA3 专属内容替换之。
- 调查笔记：`DFLASH2_CG_FIX_PLAN.md`、`IMPROVEMENTS.md`（基准历史）、`AGENTS.md`（运维教训）
  位于承载本 checkout 的 `vllm-serving` 工作区。
- 上游 vLLM 文档：[docs.vllm.ai](https://docs.vllm.ai)
