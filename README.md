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

## Building from source

```bash
# Prerequisites: ROCm >= 6.x with a matching PyTorch ROCm wheel, CMake >= 3.26, HIP toolchain.
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

## Hardware & software environment

| Item | Value |
|---|---|
| GPUs | 2× AMD Radeon RX 7900 XTX, 24 GiB each, gfx1100, TP=2 |
| OS | Linux (ROCm-enabled), see upstream [ROCm install guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| ROCm env | `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `HSA_ENABLE_IPC_MODE_LEGACY=0`, `HSA_FORCE_FINE_GRAIN_AMDGPU=1`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `GCN_ARCH_NAME=gfx1100`, `VLLM_ROCM_USE_AITER=1` |
| Target model | Qwen3.8-27B (W4A16 GPTQ, hidden 5120), FP8 KV cache |
| DFlash2 drafter | 5-layer `Qwen3.8-27B-DFlash2-bf16`, `num_speculative_tokens=7`, sliding window 2048 |

> **Note**: `HSA_ENABLE_IPC_MODE_LEGACY=0` and `HSA_FORCE_FINE_GRAIN_AMDGPU=1` are **required**
> for TP=2 stability — dropping them caused a `ProcessGroupNCCL` crash on the first request.
> They are not referenced in vLLM source but are read by the HIP runtime.

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
`-:.*mtp.*`). DFlash2 uses the same checkpoint as the target; the DFlash2 drafter itself is a
separate checkpoint, served locally as `Qwen3.8-27B-DFlash2-bf16` and mirrored upstream at
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).

---

## How to run

### 1. DFlash2 speculative decoding (default)

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

### 2. Alternatives

- **No speculative decoding**: omit `--speculative-config`.
- **MTP** (model-native heads, K=3): `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  plus `VLLM_USE_V2_MODEL_RUNNER=1` (see `run-vllm-mtp3-v2.sh`). Note MTP currently has an
  intermittent long-context NCCL watchdog hang on this stack (3 runs, 1 passed).

---

## Performance numbers

Measured with `llama-benchy 0.4.0` (`--pp 2048 --tg 128 --concurrency 1`, `2025-08-23`).
Cells are `prefill (PP, tok/s) / decode (TG, tok/s)`. DFlash2/MTP are fair reruns without
`VLLM_PROFILE_STEP`; stock vLLM and llama.cpp are from the same benchmark series.

| Context depth | stock vLLM (no spec) | **vLLM DFlash2 (K=7)** | vLLM MTP (K=3) | llama.cpp (Q6_K_XL, MTP) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1885 / 52.4 | 1805 / **93.1** | 1948 / **94.3** | 803 / 63.7 |
| **8k** | 1657 / 46.9 | 1662 / **79.8** | 1646 / **86.8** | 929 / 65.6 |
| **16k** | 1449 / 43.6 | 1447 / **69.1** | 1449 / 68.8 | 904 / 58.9 |
| **32k** | — | 1163 / **62.0** | 1190 / **64.2** | — |
| **64k** | — | 836 / **56.1** | 848 / **56.9** | — |

### Takeaways

- **Speculative decoding wins big**: DFlash2/MTP beat stock vLLM decode by **+30–80%** across
  0–64k context (52→93 tok/s at depth 0).
- **DFlash2 vs MTP are statistically tied** (differences within ±10% measurement noise);
  the only stable gap is depth 8k where MTP leads ~10%.
- **Depth decay is identical** for both (~-40% at 64k, 93→56 tok/s).
- **KV cache trade-off**: DFlash2's 5-layer drafter needs its own KV cache → usable cache is
  **7.9 GiB / 360k tokens** vs MTP's **10.5 GiB / 592k tokens** (-39%), so max long-context
  concurrency is ~1.4x vs 2.3x at 262k max_model_len.
- **llama.cpp reference** (UD-Q6_K_XL, Q8_0 KV, MTP): better per-request decode under
  concurrency (c2/c4), but slower single-stream prefill (803–929 vs 1449–1948 tok/s) and
  no 32k/64k numbers in this series.

### Measurement notes

- Single-run variance is high (±10–20%) due to Triton JIT spikes and GPU clock state;
  DFlash2 numbers are runs=2 means, MTP is a single successful run (3rd attempt).

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
