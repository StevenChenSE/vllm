# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

import triton
import triton.language as tl

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


@triton.jit
def _grouped_conv_fused_kernel(
    hidden_ptr,
    delta_ptr,
    base_ptr,
    output_ptr,
    N: tl.int32,
    stride_delta_n: tl.int32,
    stride_delta_t: tl.int32,
    stride_delta_g: tl.int32,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_e = tl.arange(0, group_size)

    # Base weights for tap 0 and tap 1 for this group
    base_idx_0 = 0 * hidden_size + pid_g * group_size + offs_e
    base_idx_1 = 1 * hidden_size + pid_g * group_size + offs_e

    base_0 = tl.load(base_ptr + base_idx_0).to(tl.float32)
    base_1 = tl.load(base_ptr + base_idx_1).to(tl.float32)

    # Delta weights: [N, taps, num_groups] with dynamic strides
    delta_idx_0 = offs_n * stride_delta_n + 0 * stride_delta_t + pid_g * stride_delta_g
    delta_idx_1 = offs_n * stride_delta_n + 1 * stride_delta_t + pid_g * stride_delta_g

    delta_0 = tl.load(delta_ptr + delta_idx_0, mask=mask_n, other=0.0).to(tl.float32)
    delta_1 = tl.load(delta_ptr + delta_idx_1, mask=mask_n, other=0.0).to(tl.float32)

    coeff_0 = base_0[None, :] + delta_0[:, None]
    coeff_1 = base_1[None, :] + delta_1[:, None]

    # Current hidden states: [BLOCK_N, group_size]
    h_idx_0 = offs_n[:, None] * hidden_size + (pid_g * group_size + offs_e[None, :])
    h_0 = tl.load(hidden_ptr + h_idx_0, mask=mask_n[:, None], other=0.0).to(tl.float32)

    out = coeff_0 * h_0

    # Tap 1 (shifted by 1 if pos % block_size >= 1)
    pos = offs_n % block_size
    valid_tap1 = mask_n & (pos >= 1)

    prev_n = tl.maximum(offs_n - 1, 0)
    h_idx_1 = prev_n[:, None] * hidden_size + (pid_g * group_size + offs_e[None, :])
    h_1 = tl.load(hidden_ptr + h_idx_1, mask=valid_tap1[:, None], other=0.0).to(tl.float32)

    out += coeff_1 * h_1

    tl.store(
        output_ptr + h_idx_0,
        out.to(output_ptr.dtype.element_ty),
        mask=mask_n[:, None],
    )


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    N = hidden_states.shape[0]
    hidden_size = num_groups * group_size
    if (
        taps == 2
        and hidden_states.is_cuda
        and hidden_states.shape[-1] == hidden_size
        and hidden_states.is_contiguous()
        and base.is_contiguous()
    ):
        output = torch.empty_like(hidden_states)
        BLOCK_N = 32
        grid = (triton.cdiv(N, BLOCK_N), num_groups)
        _grouped_conv_fused_kernel[grid](
            hidden_states,
            delta,
            base,
            output,
            N,
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            num_groups=num_groups,
            group_size=group_size,
            hidden_size=hidden_size,
            block_size=block_size,
            BLOCK_N=BLOCK_N,
        )
        return output

    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.layer_idx = layer_idx
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=speculative_config.draft_model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused Attention Branch
        if residual is None:
            residual = hidden_states
            normed_h = self.input_layernorm(hidden_states)
        else:
            normed_h, residual = self.input_layernorm(hidden_states, residual)

        conv_h, coefficients = self.attention_conv.prepare(normed_h)
        attn_out = self.self_attn(positions=positions, hidden_states=conv_h)
        hidden_states = self.attention_conv.finish(attn_out, coefficients)

        # Fused MLP Branch
        normed_h, residual = self.post_attention_layernorm(hidden_states, residual)
        conv_mlp_in, mlp_coeff = self.mlp_conv.prepare(normed_h)
        mlp_out = self.mlp(conv_mlp_in)
        hidden_states = self.mlp_conv.finish(mlp_out, mlp_coeff)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    vocab_size = predecessor_table.shape[0]
    cand_valid = (candidate_ids >= 0) & (candidate_ids < vocab_size)
    anchor_valid = (anchor_token_ids >= 0) & (anchor_token_ids < vocab_size)
    safe_cand_ids = torch.clamp(candidate_ids, 0, vocab_size - 1)
    safe_anchor_ids = torch.clamp(anchor_token_ids, 0, vocab_size - 1)
    top_k = candidate_ids.shape[-1]
    successors = successor_table[safe_cand_ids]
    predecessor_ids = torch.cat(
        (
            safe_anchor_ids[:, None, None].expand(-1, 1, top_k),
            safe_cand_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    # Fast batched GEMM: (B * L, K, R) x (B * L, R, K) -> (B * L, K, K)
    B, L, K, R = predecessors.shape
    cond = (predecessors * hidden.unsqueeze(2)).view(B * L, K, R)
    succ_t = successors.view(B * L, K, R).transpose(1, 2)
    scores = torch.bmm(cond, succ_t).view(B, L, K, K)
    total_scores = unary_logits.unsqueeze(2) + scores
    pred_valid = torch.cat(
        (
            anchor_valid[:, None, None].expand(-1, 1, top_k),
            cand_valid[:, :-1],
        ),
        dim=1,
    )
    valid_mask = pred_valid.unsqueeze(3) & cand_valid.unsqueeze(2)
    return total_scores.masked_fill(~valid_mask, -float("inf"))


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.speculative_config.draft_model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
