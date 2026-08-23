<div align="center">

# 双 AMD Radeon RX 7900 XTX (RDNA3) 上的 vLLM

[English](README.md) | [简体中文](README.zh-CN.md)

[vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork，包含 RDNA3/gfx1100 平台性能工作：
DFlash2 drafter 修复、ROCm 内核优化，以及调优过的双 7900 XTX 服务配置。

</div>

---

## 关于本分支

本分支（`feat/dflash-perf-opt`）在上游 vLLM 之上携带 RDNA3 专用改进：

- **DFlash2 投机解码修复** — 解决 CUDA Graph 下的零接受率塌方问题（ROCm 平台）。
  接受率已恢复、draft token 真实、0–64k 上下文稳定。
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

## 从源码构建

```bash
# 前置：ROCm >= 6.x + 匹配的 PyTorch ROCm wheel，CMake >= 3.26，HIP 工具链。
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

## 硬件与软件环境

| 项目 | 配置 |
|---|---|
| 显卡 | 2× AMD Radeon RX 7900 XTX，各 24 GiB，gfx1100，TP=2 |
| 系统 | Linux（启用 ROCm），见上游 [ROCm 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| ROCm 环境变量 | `HSA_OVERRIDE_GFX_VERSION=11.0.0`、`HSA_ENABLE_IPC_MODE_LEGACY=0`、`HSA_FORCE_FINE_GRAIN_AMDGPU=1`、`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`、`GCN_ARCH_NAME=gfx1100`、`VLLM_ROCM_USE_AITER=1` |
| 目标模型 | Qwen3.8-27B（W4A16 GPTQ，hidden 5120），FP8 KV cache |
| DFlash2 drafter | 5 层 `Qwen3.8-27B-DFlash2-bf16`，`num_speculative_tokens=7`，sliding window 2048 |

> **注意**：`HSA_ENABLE_IPC_MODE_LEGACY=0` 与 `HSA_FORCE_FINE_GRAIN_AMDGPU=1` 对 TP=2 稳定性
> **必需** —— 去掉后第一个请求即触发 `ProcessGroupNCCL` 崩溃。它们不出现在 vLLM 源码中，
> 但由 HIP 运行时读取。

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
同一 checkpoint 作为目标模型；DFlash2 drafter 本身是独立 checkpoint，本地以
`Qwen3.8-27B-DFlash2-bf16` 提供，上游镜像见
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)。

---

## 如何运行

### 1. DFlash2 投机解码（默认）

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
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"dflash","model":"/path/to/Qwen3.8-27B-DFlash2-bf16","num_speculative_tokens":7}' \
  --compilation-config.cudagraph_capture_sizes "[1, 2, 4, 8]"
```

启动后请发送一次 warm-up 请求以吸收首次 Triton JIT 编译尖峰（否则前几个基准深度会低测
约 20%）。公平基准必须关闭 `VLLM_PROFILE_STEP` —— 其每步 sync+print 会同时拖慢 prefill
与 decode 测量。

### 2. 其他方案

- **无投机解码**：省略 `--speculative-config`。
- **MTP**（模型自带头，K=3）：`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  并加 `VLLM_USE_V2_MODEL_RUNNER=1`（见 `run-vllm-mtp3-v2.sh`）。注意 MTP 在本平台长上下文有
  偶发的 NCCL watchdog hang（3 次实测仅 1 次通过）。

---

## 性能数据

使用 `llama-benchy 0.4.0`（`--pp 2048 --tg 128 --concurrency 1`，2025-08-23）实测。
单元格为 `prefill (PP, tok/s) / decode (TG, tok/s)`。DFlash2/MTP 为关闭 `VLLM_PROFILE_STEP`
后的公平复测；stock vLLM 与 llama.cpp 来自同一基准系列。

| 上下文深度 | stock vLLM（无投机） | **vLLM DFlash2 (K=7)** | vLLM MTP (K=3) | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1885 / 52.4 | 1805 / **93.1** | 1948 / **94.3** | 803 / 63.7 |
| **8k** | 1657 / 46.9 | 1662 / **79.8** | 1646 / **86.8** | 929 / 65.6 |
| **16k** | 1449 / 43.6 | 1447 / **69.1** | 1449 / 68.8 | 904 / 58.9 |
| **32k** | — | 1163 / **62.0** | 1190 / **64.2** | — |
| **64k** | — | 836 / **56.1** | 848 / **56.9** | — |

### 要点

- **投机解码收益显著**：0–64k 全深度 DFlash2/MTP 的 decode 领先 stock vLLM **30–80%**
  （depth 0 从 52→93 tok/s）。
- **DFlash2 与 MTP 统计持平**（差异在 ±10% 测量噪声内）；唯一稳定差异是 8k 深度 MTP
  领先约 10%。
- **两者深度衰减一致**（64k 时约 -40%，93→56 tok/s）。
- **KV cache 权衡**：DFlash2 的 5 层 drafter 需要自己的 KV cache → 可用缓存为
  **7.9 GiB / 36 万 tokens**，而 MTP 为 **10.5 GiB / 59 万 tokens**（少 39%），因此在
  262k max_model_len 下最大长上下文并发约 1.4x vs 2.3x。
- **llama.cpp 参考**（UD-Q6_K_XL, Q8_0 KV, MTP）：并发（c2/c4）下单请求 decode 更好，
  但单流 prefill 较慢（803–929 vs 1449–1948 tok/s），且本系列无 32k/64k 数据。

### 测量说明

- 单次测量波动 ±10–20%（Triton JIT 尖峰与 GPU 时钟状态）；DFlash2 为 runs=2 均值，
  MTP 为第 3 次成功运行的数值。

---

## 关键发现与坑

1. **`p_min` 提前截断：短上下文有益、长上下文有害** —— `DFLASH_P_MIN=0.3` 在 depth 0
   +9% decode，但在 8k 深度最多 -11%；长上下文场景保持 0。
2. **vLLM `/health` 返回空 body + HTTP 200** —— 需检查 HTTP code 而非 body。
3. **不要用 `pkill -9 -f "vllm"`** —— 会匹配到执行命令的 bash 自身，同一行后续命令静默
   跳过；改用 `rocm-smi --showpids` 取 PID 杀进程。

---

## 参考

- 上游原始 README：[vllm-project/vllm](https://github.com/vllm-project/vllm#readme) —— 本 fork
  以 RDNA3 专属内容替换之。
- 调查笔记：`DFLASH2_CG_FIX_PLAN.md`、`IMPROVEMENTS.md`（基准历史）、`AGENTS.md`（运维教训）
  位于承载本 checkout 的 `vllm-serving` 工作区。
- 上游 vLLM 文档：[docs.vllm.ai](https://docs.vllm.ai)
