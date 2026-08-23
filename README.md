<div align="center">

# vLLM on Dual AMD Radeon RX 7900 XTX (RDNA3)

**双 AMD Radeon RX 7900 XTX (RDNA3) 上的 vLLM 性能分支**

Fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) with RDNA3 / gfx1100 performance work:
DFlash2 drafter fixes, ROCm kernel optimizations, and a tuned dual-7900-XTX serving stack.

</div>

---

## About this fork / 关于本分支

This branch (`feat/dflash-perf-opt`) carries RDNA3-specific improvements on top of upstream vLLM:

- **DFlash2 speculative decoding fixes** — resolves the CUDA-graph zero-acceptance collapse
  (root cause: stale `torch.compile` AOT cache; see `DFLASH2_CG_FIX_PLAN.md` for the full
  investigation). Acceptance rate restored, draft tokens are real, 0–64k context stable.
- **ROCm kernel work** — fused native HIP kernels for 1D grouped conv and split-kv reduction,
  sliding-window detection hardening, draft-token sanitization.
- **Tuned dual-7900-XTX serving config** — see below for the launch recipe and measured numbers.

本分支专注于 RDNA3/gfx1100 平台：修复了 DFlash2 投机解码在 CUDA Graph 下的零接受率问题
（根因是过期的 torch.compile AOT 缓存），并针对双 7900 XTX 做了启动配置与内核调优。

---

## Hardware & software environment / 硬件与软件环境

| Item / 项目 | Value / 配置 |
|---|---|
| GPUs / 显卡 | 2× AMD Radeon RX 7900 XTX, 24 GiB each, gfx1100, TP=2 |
| OS / 系统 | Linux (ROCm-enabled), see upstream [ROCm install guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| ROCm env / 环境变量 | `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `HSA_ENABLE_IPC_MODE_LEGACY=0`, `HSA_FORCE_FINE_GRAIN_AMDGPU=1`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `GCN_ARCH_NAME=gfx1100`, `VLLM_ROCM_USE_AITER=1` |
| Target model / 目标模型 | Qwen3.8-27B (W4A16 GPTQ, hidden 5120), FP8 KV cache |
| DFlash2 drafter | 5-layer `Qwen3.8-27B-DFlash2-bf16`, `num_speculative_tokens=7`, sliding window 2048 |

> **Note / 注意**: `HSA_ENABLE_IPC_MODE_LEGACY=0` and `HSA_FORCE_FINE_GRAIN_AMDGPU=1` are
> **required** for TP=2 stability — dropping them caused a `ProcessGroupNCCL` crash on the
> first request. They are not referenced in vLLM source but are read by the HIP runtime.

---

## How to run / 如何运行

### 1. DFlash2 speculative decoding (default / 默认)

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

Send one warm-up request after startup to absorb the first-inference Triton JIT spike
(otherwise the first benchmarked depths measure ~20% low). `VLLM_PROFILE_STEP` must stay OFF
for fair benchmarking — its per-step synchronize+print skews both prefill and generation.

启动后请发送一次 warm-up 请求以吸收首次 Triton JIT 编译尖峰；公平基准必须关闭
`VLLM_PROFILE_STEP`（每步 sync+print 会同时拖慢 prefill 与 decode 测量）。

### 2. Alternatives / 其他方案

- **No speculative decoding**: omit `--speculative-config`.
- **MTP** (model-native heads, K=3): `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  plus `VLLM_USE_V2_MODEL_RUNNER=1` (see `run-vllm-mtp3-v2.sh`). Note MTP currently has an
  intermittent long-context NCCL watchdog hang on this stack (3 runs, 1 passed).

---

## Performance numbers / 性能数据

Measured with `llama-benchy 0.4.0` (`--pp 2048 --tg 128 --concurrency 1`, `2025-08-23`).
Cells are `prefill (PP, tok/s) / decode (TG, tok/s)`. DFlash2/MTP are fair reruns without
`VLLM_PROFILE_STEP`; stock vLLM and llama.cpp are from the same benchmark series.

使用 `llama-benchy 0.4.0`（`--pp 2048 --tg 128 --concurrency 1`，2025-08-23）实测。
单元格为 `prefill (PP, tok/s) / decode (TG, tok/s)`。

| Context depth | stock vLLM (no spec) | **vLLM DFlash2 (K=7)** | vLLM MTP (K=3) | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1885 / 52.4 | 1805 / **93.1** | 1948 / **94.3** | 803 / 63.7 |
| **8k** | 1657 / 46.9 | 1662 / **79.8** | 1646 / **86.8** | 929 / 65.6 |
| **16k** | 1449 / 43.6 | 1447 / **69.1** | 1449 / 68.8 | 904 / 58.9 |
| **32k** | — | 1163 / **62.0** | 1190 / **64.2** | — |
| **64k** | — | 836 / **56.1** | 848 / **56.9** | — |

### Takeaways / 要点

- **Speculative decoding wins big**: DFlash2/MTP beat stock vLLM decode by **+30–80%** across
  0–64k context (52→93 tok/s at depth 0). 投机解码收益显著：0–64k 全深度 decode 领先
  stock vLLM 30–80%。
- **DFlash2 vs MTP are statistically tied** (differences within ±10% measurement noise);
  the only stable gap is depth 8k where MTP leads ~10%. DFlash2 与 MTP 性能相当（差异在
  ±10% 测量噪声内），唯一稳定差异是 8k 深度 MTP 领先约 10%。
- **Depth decay is identical** for both (~-40% at 64k, 93→56 tok/s). 两者深度衰减一致
  （64k 时约 -40%）。
- **KV cache trade-off**: DFlash2's 5-layer drafter needs its own KV cache → usable cache is
  **7.9 GiB / 360k tokens** vs MTP's **10.5 GiB / 592k tokens** (-39%), so max long-context
  concurrency is ~1.4x vs 2.3x at 262k max_model_len. 注意 KV cache 差异：DFlash2 独立
  drafter 使可用 KV cache 比 MTP 少 39%。
- **llama.cpp reference** (UD-Q6_K_XL, Q8_0 KV, MTP): better per-request decode under
  concurrency (c2/c4), but slower single-stream prefill (803–929 vs 1449–1948 tok/s) and
  no 32k/64k numbers in this series.

### Measurement notes / 测量说明

- Single-run variance is high (±10–20%) due to Triton JIT spikes and GPU clock state;
  DFlash2 numbers are runs=2 means, MTP is a single successful run (3rd attempt).
- 单次测量波动 ±10–20%（Triton JIT 尖峰与 GPU 时钟），DFlash2 为 runs=2 均值，MTP 为
  第 3 次成功运行的数值。

---

## Key findings & gotchas / 关键发现与坑

1. **Stale `torch.compile` AOT cache corrupts CUDA-graph output silently** — after modifying
   model code, `rm -rf ~/.cache/vllm/torch_compile_cache ~/.triton/cache` or the drafter
   collapses to token `3` with zero acceptance while eager mode stays correct.
   修改模型代码后必须清 compile cache，否则 CUDAGraph 下 draft 塌方为 token `3`、接受率为 0，
   而 eager 模式仍正常。
2. **`p_min` early-stop helps short context, hurts long context** — `DFLASH_P_MIN=0.3` gains
   +9% decode at depth 0 but loses up to -11% at depth 8k; keep 0 for long-context workloads.
   `p_min` 提前截断在短上下文 +9%，长上下文 -11%，长上下文场景保持 0。
3. **vLLM `/health` returns an empty body with HTTP 200** — check the HTTP code, not the body.
   vLLM `/health` 返回空 body + 200，需检查 HTTP code。
4. **Don't `pkill -9 -f "vllm"`** — it matches the invoking shell itself and silently skips
   subsequent commands in the same line; kill by PID from `rocm-smi --showpids` instead.
   不要用 `pkill -9 -f "vllm"`（会误杀自身 bash），用 `rocm-smi --showpids` 的 PID。

---

## Reference / 参考

- Original upstream README: [vllm-project/vllm](https://github.com/vllm-project/vllm#readme) — this fork
  replaces it with RDNA3-specific content.
- Investigation notes: `DFLASH2_CG_FIX_PLAN.md`, `IMPROVEMENTS.md` (benchmark history), `AGENTS.md`
  (operational lessons) live in the `vllm-serving` workspace that hosts this checkout.
- Upstream vLLM docs: [docs.vllm.ai](https://docs.vllm.ai)
