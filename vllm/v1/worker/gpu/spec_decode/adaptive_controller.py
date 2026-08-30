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

        if num_sampled.numel() == 0:
            return

        idx_map = idx_mapping.to(device=self.device, dtype=torch.int64)
        valid_req_mask = (idx_map >= 0) & (idx_map < self.max_num_reqs)
        safe_req_indices = torch.where(valid_req_mask, idx_map, 0)

        last_draft = self.last_draft_len[safe_req_indices]
        sampled_cnt = num_sampled.to(device=self.device, dtype=torch.float32)

        # Update condition: valid slot index, had active drafts in previous step, and non-zero sampled (not chunked prefill)
        update_mask = valid_req_mask & (last_draft > 0) & (sampled_cnt > 0)

        target_indices = safe_req_indices[update_mask]
        if target_indices.numel() == 0:
            return

        n_accepted = torch.clamp(sampled_cnt[update_mask] - 1.0, min=0.0)
        n_drafted = last_draft[update_mask].to(torch.float32)
        current_ema = self.acc_ema[target_indices]

        # Full acceptance: Censored observation -> additive probe (+probe_step) capped at n_max
        probed_ema = torch.clamp(current_ema + self.probe_step, max=float(self.n_max))
        # Partial acceptance: Uncensored observation -> EMA decay
        decayed_ema = (1.0 - self.alpha) * current_ema + self.alpha * n_accepted

        censored_mask = n_accepted >= n_drafted
        new_ema = torch.where(censored_mask, probed_ema, decayed_ema)

        self.acc_ema[target_indices] = new_ema
        self.last_draft_len[target_indices] = 0

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
