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

### 1. DFlash2 speculative decoding (BF16 Drafter)

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

### 2. DFlash2 speculative decoding (W4A16 Quantized Drafter: `syvai/Qwen3.8-27B-DFlash2-W4A16`)

Saves ~0.8 GiB VRAM per GPU and boosts mathematical reasoning decode to **182.9 tok/s** (peak **197.3 tok/s** on MATH-500):

```bash
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

### 3. MTP3 speculative decoding (Native MTP Heads, `align` mode)

Best for ultra long-context agentic workloads and high concurrency (shares target KV cache, **11.3 GiB / 638k tokens**):

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

> **Note**: Send one warm-up request after startup to absorb the first-inference Triton JIT spike. `VLLM_PROFILE_STEP` must stay OFF for fair benchmarking — its per-step synchronize+print skews both prefill and generation.

---

## Performance numbers

Measured with `llama-benchy 0.4.0` (`--pp 2048 --tg 128 --concurrency 1`, updated post upstream merge on `2026-08-27`).
Cells are `prefill (PP, tok/s) / decode (TG, tok/s)`. DFlash2/MTP are fair runs without `VLLM_PROFILE_STEP`.

| Context depth | stock vLLM (upstream, early) | **vLLM DFlash2 (BF16)** | **vLLM DFlash2 (W4A16)** | **vLLM MTP3 (BF16, align)** | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1773 / 53.8 | 1634 / **98.7** (peak 107.0) | 1666 / **101.9** (peak 107.0) | 1617 / **102.3** (peak 105.0) | 803 / 63.7 |
| **4k** | — | 1742 / **89.5** (peak 92.0) | 1752 / **76.7** (peak 78.0) | 1696 / **96.5** (peak 107.0) | — |
| **8k** | 1131 / 49.4 | 1673 / **81.6** (peak 83.0) | 1684 / **66.9** (peak 74.0) | 1592 / **102.0** (peak 110.0) | 929 / 65.6 |
| **16k** | 838 / 45.7 | 1495 / **82.2** (peak 90.0) | 1488 / **71.7** (peak 73.0) | 1416 / **90.5** (peak 101.0) | 904 / 58.9 |
| **32k** | — | 1163 / **62.0** | — | 1190 / **64.2** | — |
| **64k** | — | 836 / **56.1** | — | 848 / **56.9** | — |

### Takeaways

- **Speculative decoding wins big**: DFlash2/MTP beat stock vLLM decode by **+50–90%** across 0–16k context (54→98~102 tok/s at depth 0).
- **MTP3 leads in medium-to-deep context**: MTP3 sustains **102 tok/s @ 8k** and **90.5 tok/s @ 16k**, outperforming DFlash2 in deep contexts while sharing target KV cache.
- **DFlash2 dominates structured math/code generation**: DFlash2-W4A16 reaches **182.9 tok/s** (peak 197.3 tok/s on MATH-500) and DFlash2-BF16 reaches **178.2 tok/s**, outperforming MTP3 (131.4 tok/s) by **+35–39%**.
- **KV cache trade-off**:
  - **MTP3**: **11.3 GiB / 638k tokens** usable KV cache (~2.4x concurrency @ 262k).
  - **DFlash2 (W4A16)**: **7.7 GiB / 335k tokens** usable KV cache (~1.68x concurrency @ 200k).
  - **DFlash2 (BF16)**: **6.5 GiB / 282k tokens** usable KV cache (~1.41x concurrency @ 200k).

---

## Key performance claims

**Math reasoning decode @ depth 0** (GSM8K + MATH-500, greedy, 4 problems, 100% accuracy):

| engine | avg decode |
| :--- | :---: |
| vLLM Baseline (no spec) | 56.3 tok/s |
| **vLLM DFlash2 (W4A16, K=7)** | **182.9 tok/s (3.24×)** 🏆 |
| **vLLM DFlash2 (BF16, K=7)** | **178.2 tok/s (3.16×)** |
| vLLM MTP3 (K=3) | 131.4 tok/s (2.33×) |
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

## Reference

- Original upstream README: [vllm-project/vllm](https://github.com/vllm-project/vllm#readme) — this fork
  replaces it with RDNA3-specific content.
- Investigation notes: `DFLASH2_CG_FIX_PLAN.md`, `IMPROVEMENTS.md` (benchmark history), `AGENTS.md`
  (operational lessons) live in the `vllm-serving` workspace that hosts this checkout.
- Upstream vLLM docs: [docs.vllm.ai](https://docs.vllm.ai)
