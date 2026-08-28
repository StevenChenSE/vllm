# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive Speculative Decoding Controller.

Based on online censored-observation acceptance tracking:
- Full acceptance (n_acc >= n_draft): Censored lower bound -> additive probe (+1.0)
- Partial acceptance (n_acc < n_draft): Uncensored observation -> EMA decay (alpha=0.25)
- Dynamic floor (n_min) and ceiling (n_max) clamping.
"""

from typing import Optional
import torch


class AdaptiveSpecController:
    """Per-sequence online controller for adaptive speculative draft length."""

    def __init__(
        self,
        max_num_reqs: int,
        n_max: int,
        n_min: int = 1,
        alpha: float = 0.25,
        probe_step: float = 1.0,
        init_val: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        enabled: bool = True,
    ):
        self.max_num_reqs = max_num_reqs
        self.n_max = n_max
        self.n_min = max(1, n_min)
        self.alpha = alpha
        self.probe_step = probe_step
        self.init_val = init_val if init_val is not None else float(self.n_max)
        self.device = device
        self.enabled = enabled

        # EMA state per request slot
        self.acc_ema = torch.full(
            (max_num_reqs,), self.init_val, dtype=torch.float32, device=device
        )
        # Size of the draft issued in the previous step
        self.last_draft_len = torch.zeros(
            (max_num_reqs,), dtype=torch.int32, device=device
        )

    def reset_request(self, req_state_idx: int) -> None:
        """Reset tracking state for a new sequence or recycled request slot."""
        if 0 <= req_state_idx < self.max_num_reqs:
            self.acc_ema[req_state_idx] = self.init_val
            self.last_draft_len[req_state_idx] = 0

    def reset_requests(self, req_state_indices: torch.Tensor) -> None:
        """Batch reset for newly added/cleared request indices."""
        valid_mask = (req_state_indices >= 0) & (req_state_indices < self.max_num_reqs)
        valid_indices = req_state_indices[valid_mask]
        if valid_indices.numel() > 0:
            self.acc_ema[valid_indices] = self.init_val
            self.last_draft_len[valid_indices] = 0

    def update_acceptance(
        self,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        """Update EMA of accepted tokens based on post-verification results.

        Args:
            num_sampled: [num_reqs] count of accepted tokens (1 bonus + accepted drafts)
            num_rejected: [num_reqs] count of rejected draft tokens
            idx_mapping: [num_reqs] request state index mapping
        """
        if not self.enabled:
            return

        num_reqs = num_sampled.shape[0]
        for i in range(num_reqs):
            req_idx = int(idx_mapping[i].item())
            if req_idx < 0 or req_idx >= self.max_num_reqs:
                continue

            n_drafted = int(self.last_draft_len[req_idx].item())
            if n_drafted <= 0:
                continue

            sampled_cnt = int(num_sampled[i].item())
            if sampled_cnt == 0:
                # Chunked prefill step: skip updating decode acceptance EMA
                continue

            n_accepted = max(0, sampled_cnt - 1)
            current_ema = float(self.acc_ema[req_idx].item())

            if n_accepted >= n_drafted:
                # Censored observation: true predictability is at least n_drafted -> probe upward
                new_ema = min(float(self.n_max), current_ema + self.probe_step)
            else:
                # Uncensored observation: exact stopping point observed -> EMA decay
                new_ema = (1.0 - self.alpha) * current_ema + self.alpha * float(n_accepted)

            self.acc_ema[req_idx] = new_ema
            self.last_draft_len[req_idx] = 0

    def get_effective_draft_length(
        self,
        req_state_idx: int,
        n_max: Optional[int] = None,
        n_min: Optional[int] = None,
    ) -> int:
        """Calculate effective draft token count for a given sequence."""
        if not self.enabled:
            return n_max if n_max is not None else self.n_max

        cur_max = n_max if n_max is not None else self.n_max
        cur_min = n_min if n_min is not None else self.n_min

        if req_state_idx < 0 or req_state_idx >= self.max_num_reqs:
            return cur_max

        ema_val = float(self.acc_ema[req_state_idx].item())
        target_len = int(round(ema_val))
        eff_len = max(cur_min, min(cur_max, target_len))

        self.last_draft_len[req_state_idx] = eff_len
        return eff_len

    def get_batch_effective_draft_lengths(
        self,
        idx_mapping: torch.Tensor,
        n_max: Optional[int] = None,
        n_min: Optional[int] = None,
    ) -> torch.Tensor:
        """Vectorized computation of effective draft lengths for a batch."""
        cur_max = n_max if n_max is not None else self.n_max
        cur_min = n_min if n_min is not None else self.n_min

        if not self.enabled:
            return torch.full(
                (idx_mapping.shape[0],), cur_max, dtype=torch.int32, device=self.device
            )

        valid_mask = (idx_mapping >= 0) & (idx_mapping < self.max_num_reqs)
        safe_indices = torch.where(valid_mask, idx_mapping, 0)
        ema_vals = self.acc_ema[safe_indices]
        eff_lens = torch.clamp(torch.round(ema_vals).to(torch.int32), cur_min, cur_max)
        eff_lens = torch.where(valid_mask, eff_lens, cur_max)

        valid_indices = safe_indices[valid_mask]
        if valid_indices.numel() > 0:
            self.last_draft_len[valid_indices] = eff_lens[valid_mask]
        return eff_lens
