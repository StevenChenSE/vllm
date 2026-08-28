# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU Triton & Vectorized N-Gram (Prompt Lookup) Matcher for Hybrid Speculative Decoding.

Inspired by llama.cpp commit 925e11799 (token ID tracking in KV cell for N-gram lookup).
Searches context token history for matching suffix N-grams (N=2..4) and extracts following tokens
as high-confidence draft candidates.
"""

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _ngram_lookup_kernel(
    all_token_ids_ptr,
    all_token_ids_stride,
    total_lens_ptr,
    idx_mapping_ptr,
    ngram_draft_ptr,
    ngram_draft_stride,
    ngram_match_len_ptr,
    num_reqs: int,
    max_spec_tokens: int,
    min_ngram: int,
    max_ngram: int,
    max_history_search: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_idx = tl.load(idx_mapping_ptr + pid)
    if req_idx < 0:
        return

    total_len = tl.load(total_lens_ptr + req_idx)
    if total_len <= min_ngram:
        tl.store(ngram_match_len_ptr + pid, 0)
        return

    token_base = all_token_ids_ptr + req_idx * all_token_ids_stride

    # Try N-gram lengths from max_ngram down to min_ngram
    best_match_pos = -1
    best_match_n = 0

    # Search window bound
    search_start = 0
    if total_len > max_history_search:
        search_start = total_len - max_history_search

    # Scan for N-gram matches
    for n in range(max_ngram, min_ngram - 1, -1):
        if total_len > n and best_match_pos < 0:
            # Trailing N tokens
            suffix_0 = tl.load(token_base + total_len - n)
            suffix_1 = tl.load(token_base + total_len - n + 1) if n >= 2 else 0
            suffix_2 = tl.load(token_base + total_len - n + 2) if n >= 3 else 0
            suffix_3 = tl.load(token_base + total_len - n + 3) if n >= 4 else 0

            # Scan history backwards (latest occurrence first)
            scan_limit = total_len - n
            for p in range(scan_limit - 1, search_start - 1, -1):
                if best_match_pos < 0:
                    m0 = tl.load(token_base + p) == suffix_0
                    m1 = (tl.load(token_base + p + 1) == suffix_1) if n >= 2 else True
                    m2 = (tl.load(token_base + p + 2) == suffix_2) if n >= 3 else True
                    m3 = (tl.load(token_base + p + 3) == suffix_3) if n >= 4 else True

                    if m0 and m1 and m2 and m3:
                        best_match_pos = p
                        best_match_n = n

    if best_match_pos >= 0 and best_match_n > 0:
        match_end = best_match_pos + best_match_n
        available = total_len - match_end
        num_extract = available
        if num_extract > max_spec_tokens:
            num_extract = max_spec_tokens

        tl.store(ngram_match_len_ptr + pid, num_extract)
        out_base = ngram_draft_ptr + pid * ngram_draft_stride
        for k in range(max_spec_tokens):
            if k < num_extract:
                tok = tl.load(token_base + match_end + k)
                tl.store(out_base + k, tok)
            else:
                tl.store(out_base + k, -1)
    else:
        tl.store(ngram_match_len_ptr + pid, 0)
        out_base = ngram_draft_ptr + pid * ngram_draft_stride
        for k in range(max_spec_tokens):
            tl.store(out_base + k, -1)


class NgramLookupModule:
    """Fast GPU-resident N-gram matcher for speculative drafting."""

    def __init__(
        self,
        max_num_reqs: int,
        max_spec_tokens: int,
        min_ngram: int = 2,
        max_ngram: int = 4,
        max_history_search: int = 4096,
        device: torch.device = torch.device("cuda"),
    ):
        self.max_num_reqs = max_num_reqs
        self.max_spec_tokens = max_spec_tokens
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.max_history_search = max_history_search
        self.device = device

        self.ngram_draft_tokens = torch.full(
            (max_num_reqs, max_spec_tokens), -1, dtype=torch.int64, device=device
        )
        self.ngram_match_lens = torch.zeros(
            (max_num_reqs,), dtype=torch.int32, device=device
        )

    def lookup(
        self,
        all_token_ids: torch.Tensor,
        total_lens: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_reqs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform parallel N-gram search across all active requests in the batch.

        Returns:
            ngram_draft: [num_reqs, max_spec_tokens] tensor of candidate token IDs
            match_lens: [num_reqs] count of valid matching tokens (0 if no match)
        """
        if num_reqs <= 0:
            return self.ngram_draft_tokens[:0], self.ngram_match_lens[:0]

        grid = (num_reqs,)
        _ngram_lookup_kernel[grid](
            all_token_ids,
            all_token_ids.stride(0),
            total_lens,
            idx_mapping,
            self.ngram_draft_tokens,
            self.ngram_draft_tokens.stride(0),
            self.ngram_match_lens,
            num_reqs=num_reqs,
            max_spec_tokens=self.max_spec_tokens,
            min_ngram=self.min_ngram,
            max_ngram=self.max_ngram,
            max_history_search=self.max_history_search,
            BLOCK_SIZE=128,
        )

        return self.ngram_draft_tokens[:num_reqs], self.ngram_match_lens[:num_reqs]
