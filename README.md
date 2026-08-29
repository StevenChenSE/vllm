<div align="center">

# vLLM on Dual AMD Radeon RX 7900 XTX (RDNA3)

[English](README.md) | [简体中文](README.zh-CN.md)

Fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) with RDNA3 / gfx1100 performance work:
DFlash2 drafter fixes, ROCm kernel optimizations, and a tuned dual-7900-XTX serving stack.

</div>

---

## About this fork

This branch (`feat/dflash-perf-opt`) carries RDNA3-specific improvements on top of upstream vLLM:

- **DFlash2 speculative decoding fixes** — resolves the CUDA-graph zero-acceptance collapse on
  ROCm. Acceptance rate restored, draft tokens are real, 0–64k context stable.
- **ROCm kernel work** — fused native HIP kernels for 1D grouped conv and split-kv reduction,
  sliding-window detection hardening, draft-token sanitization.
- **RDNA3 optimizations inherited from [JartX/vllm](https://github.com/JartX/vllm)** — after
  auditing its branches, the following items are present in this branch (verified against the
  branch history):
  - Triton attention adaptive prefill tuning (`BLOCK_M`/`num_warps` by KV length, `6cb90f545`)
  - 3D split-KV softmax NaN fix for long-context decode (`6b84c6c6c`)
  - bounds-safe Triton attention indexing + `triton_quant_kv` dtype caching (`883e982e6`)
  - RDNA3 W4A16 GPTQ kernels (`csrc/rocm/q_gemm_rdna3*.cu`, in use at runtime as
    `RDNA3W4A16LinearKernel`)
  - INT8 per-tensor KV cache (`647b3883b`, code kept; not usable on this model without
    pre-calibrated scales)
  - Note: **hostar all-reduce and WMMA paged-prefill were tried and reverted** (`4a8e2652f`,
    negative end-to-end gains), and MoE work is **not** in this branch.
  **Because of these inherited gains, the "stock vLLM" column in the benchmarks below is the
  same fork with speculative decoding disabled — its prefill/decode are higher than pure
  upstream vLLM.** Credit for these improvements goes to JartX's work.
- **Tuned dual-7900-XTX serving config** — see below for the launch recipe and measured numbers.

---

## Hardware & software environment

| Item | Value |
|---|---|
| GPUs | 2× AMD Radeon RX 7900 XTX, 24 GiB each, gfx1100, TP=2 |
| ROCm | **7.14** (HIP 7.14.60850) |
| GPU driver | **amdgpu kernel module 3.2.340** (kernel 6.17.0-35-generic, Ubuntu 24.04, `amdgpu-install` 30.30.4) |
| PyTorch | 2.11.0+git, ROCm wheel (`torch.version.hip = 7.2.53211`) |
| OS | Linux (ROCm-enabled), see upstream [ROCm install guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| ROCm env | `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `HSA_ENABLE_IPC_MODE_LEGACY=0`, `HSA_FORCE_FINE_GRAIN_AMDGPU=1`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `GCN_ARCH_NAME=gfx1100`, `VLLM_ROCM_USE_AITER=1` |
| Target model | Qwen3.8-27B (W4A16 GPTQ, hidden 5120), FP8 KV cache |
| DFlash2 drafter | 5-layer `Qwen3.8-27B-DFlash2-bf16`, `num_speculative_tokens=7`, sliding window 2048 |

> **Note**: `HSA_ENABLE_IPC_MODE_LEGACY=0` and `HSA_FORCE_FINE_GRAIN_AMDGPU=1` are **required**
> for TP=2 stability — dropping them caused a `ProcessGroupNCCL` crash on the first request.
> They are not referenced in vLLM source but are read by the HIP runtime.

---

## Building from source

```bash
# Prerequisites: the ROCm/driver stack above, a matching PyTorch ROCm wheel, CMake >= 3.26, HIP toolchain.
cd vllm
pip install -e .          # auto-detects ROCm via torch.version.hip (VLLM_TARGET_DEVICE=rocm)
python -c "import vllm; print(vllm.__version__)"   # verify the editable install
```

- The HIP C++ kernels in `csrc/rocm/` (including `RDNA3W4A16LinearKernel`) are compiled by the
  same editable install; `ROCM_HOME` is picked up automatically, `HSA_OVERRIDE_GFX_VERSION=11.0.0`
  is only needed at runtime (and for Triton JIT targeting gfx1100 via `GCN_ARCH_NAME`).
- A working editable build is what the rest of this document assumes; the launch recipes use the
  repo's `serve-bootstrap.py` wrapper, but plain `vllm serve` works identically.

---

## Preparing the model checkpoint

The target uses a patched local checkpoint `qwen3.8-27b-mtp-fixed`, derived from
[`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ)
(source `config.json` verified against the live HF repo):

1. The stock repo stores all 15 MTP tensors as plain **BF16** in
   `model_extra_tensors.safetensors`, but its `quantization_config.dynamic` declares
   **positive** rules `"+:.*mtp.*"` and `"+:.*mtp\.fc.*"` (4-bit, group 64).
2. vLLM's GPTQ loader trusts that config, builds MTP layers with quantized params, and
   weight loading fails (`no module or parameter named 'layers.0.mlp.down_proj.weight'`).
3. **Patch**: replace those `"+:"` rules with a single `"-:.*mtp.*"` exclusion in
   `config.json` (`quantization_config.dynamic`). vLLM's built-in workaround
   (`qwen3_5_mtp.py`) then disables quantization for MTP layers and loads the BF16
   weights as-is. Upstream references: [vllm-project/vllm#48816](https://github.com/vllm-project/vllm/pull/48816),
   [#47828](https://github.com/vllm-project/vllm/pull/47828).

The patched `config.json` ends up with 97 negative rules and zero positive rules
(verified against the local `qwen3.8-27b-mtp-fixed/config.json`: `"mtp"` appears only as
`-:.*mtp.*`). DFlash2 uses the same checkpoint as the target; the DFlash2 drafters are
separate checkpoints:
- **BF16 Drafter**: served locally as `Qwen3.8-27B-DFlash2-bf16`, mirrored at [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).
- **W4A16 Quantized Drafter**: served locally as `Qwen3.8-27B-DFlash2-W4A16`, available at [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16).

---

## How to run

### 1. DFlash2 speculative decoding (W4A16 Quantized Drafter: `syvai/Qwen3.8-27B-DFlash2-W4A16`)

Highest peak throughput on math reasoning (**182.0 tok/s**) and agentic multi-turn sessions (**101.9 tok/s** average):

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
  --speculative-config '{"method":"dflash","model":"/path/to/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}' \
  --compilation-config.cudagraph_capture_sizes "[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]" \
  --performance-mode throughput
```

### 2. MTP3 speculative decoding (Native MTP Heads, `align` mode)

Highest generation stability (**15.8% CV jitter**) and maximum usable KV cache (**11.3 GiB / 638k tokens**):

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_IPC_MODE_LEGACY=0
export HSA_FORCE_FINE_GRAIN_AMDGPU=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export GCN_ARCH_NAME=gfx1100
export VLLM_ROCM_USE_AITER=1
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1

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
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --performance-mode throughput
```

> **Note**: Send one warm-up request after startup to absorb the first-inference Triton JIT spike. `VLLM_PROFILE_STEP` must stay OFF for fair benchmarking — its per-step synchronize+print skews both prefill and generation.

---

## Performance numbers

Measured on Dual AMD Radeon RX 7900 XTX (TP=2, gfx1100, ROCm 7.14) with fresh cache and warmup.
Values represent `prefill (PP, tok/s) / decode (TG, tok/s)`.

| Context Depth / Metric | Baseline (No Spec) | **vLLM DFlash2 (W4A16, N=7)** | **vLLM MTP3 (BF16, align, N=3)** | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: |
| **Depth 0 (Short)** | 1773 / 53.8 | 1904 / **96.9** (peak 100.1) | 1763 / **96.9** (peak 100.1) | 803 / 63.7 |
| **Depth 4k** | — | 1796 / **76.4** (peak 78.9) | 1716 / **78.9** (peak 81.5) | — |
| **Depth 8k** | 1131 / 49.4 | 1718 / **90.1** (peak 93.0) | 1640 / **83.6** (peak 86.3) | 929 / 65.6 |
| **Depth 16k** | 838 / 45.7 | 1552 / **34.2 ~ 70.0** (peak 81.0) | 1450 / **71.8 ~ 87.4** (peak 98.0) | 904 / 58.9 |
| **40k Agentic Session (Mean TG)** | 35.7 | **101.9 tok/s (peak 133.6)** 🏆 | **93.2 tok/s (peak 116.0)** | — |
| **40k Agentic Session (Jitter CV)** | — | 19.0% | **15.8% (Most Stable)** 🏆 | — |
| **Math Reasoning (GSM8K/MATH)** | 56.3 | **182.0 tok/s (peak 195.4, 3.23×)** 🏆 | **131.2 tok/s (peak 134.5, 2.33×)** | 87.5 |

### Takeaways

- **DFlash2 Baseline dominates peak decode**: Achieves **182.0 tok/s** on math reasoning (+38.7% over MTP3) and **101.9 tok/s** mean generation speed across 40k agentic sessions.
- **Tuned MTP3 excels in long-context stability**: Sustains **71.8 ~ 87.4 tok/s @ 16k** and **93.2 tok/s** across 40k sessions with the lowest generation jitter (**15.8% CV**) while sharing the target KV cache.
- **KV cache trade-off**:
  - **MTP3**: **11.3 GiB / 638k tokens** usable KV cache (~2.44x concurrency @ 262k).
  - **DFlash2 (W4A16)**: **7.7 GiB / 335k tokens** usable KV cache (~1.68x concurrency @ 200k).
  - **DFlash2 (BF16)**: **6.5 GiB / 282k tokens** usable KV cache (~1.41x concurrency @ 200k).

---

## Key performance claims

**Math reasoning decode @ depth 0** (GSM8K + MATH-500, greedy, 4 problems, 100% accuracy):

| engine | avg decode |
| :--- | :---: |
| vLLM Baseline (no spec) | 56.3 tok/s |
| **vLLM DFlash2 (W4A16, K=7)** | **182.0 tok/s (peak 195.4 tok/s, 3.23×)** 🏆 |
| **vLLM DFlash2 (BF16, K=7)** | **178.2 tok/s (3.16×)** |
| vLLM MTP3 (K=3) | 131.2 tok/s (2.33×) |
| llama.cpp (Q6_K_XL, MTP) | 87.5 tok/s (1.55×) |

(DFlash2 row measured in eager mode; full table in `IMPROVEMENTS.md` §9.2.)

---

## Key findings & gotchas

1. **`p_min` early-stop helps short context, hurts long context** — `DFLASH_P_MIN=0.3` gains
   +9% decode at depth 0 but loses up to -11% at depth 8k; keep 0 for long-context workloads.
2. **vLLM `/health` returns an empty body with HTTP 200** — check the HTTP code, not the body.
3. **Don't `pkill -9 -f "vllm"`** — it matches the invoking shell itself and silently skips
   subsequent commands in the same line; kill by PID from `rocm-smi --showpids` instead.

---

## Reference & Acknowledgements

- Original upstream README: [vllm-project/vllm](https://github.com/vllm-project/vllm#readme) — this fork
  replaces it with RDNA3-specific content.
- Investigation notes: `DFLASH2_CG_FIX_PLAN.md`, `IMPROVEMENTS.md` (benchmark history), `AGENTS.md`
  (operational lessons) live in the `vllm-serving` workspace that hosts this checkout.
- Upstream vLLM docs: [docs.vllm.ai](https://docs.vllm.ai)
