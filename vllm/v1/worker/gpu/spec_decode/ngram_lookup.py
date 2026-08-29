# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU Triton Vectorized N-Gram (Prompt Lookup) Matcher for Hybrid Speculative Decoding.

Inspired by llama.cpp commit 925e11799 (token ID tracking in KV cell for N-gram lookup).
Searches context token history using parallel SIMD block matching (N=2..4) and extracts
following tokens as high-confidence draft candidates in ~20 microseconds.
"""

from typing import Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _vectorized_ngram_lookup_kernel(
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

    out_base = ngram_draft_ptr + pid * ngram_draft_stride
    req_idx = tl.load(idx_mapping_ptr + pid)
    if req_idx < 0:
        tl.store(ngram_match_len_ptr + pid, 0)
        for k in range(max_spec_tokens):
            tl.store(out_base + k, -1)
        return

    total_len = tl.load(total_lens_ptr + req_idx)
    if total_len <= min_ngram:
        tl.store(ngram_match_len_ptr + pid, 0)
        for k in range(max_spec_tokens):
            tl.store(out_base + k, -1)
        return

    token_base = all_token_ids_ptr + req_idx * all_token_ids_stride

    search_start = 0
    if total_len > max_history_search:
        search_start = total_len - max_history_search

    best_match_pos = -1
    best_match_n = 0

    # Test N-gram from max down to min (e.g. 4, 3, 2)
    for n in range(max_ngram, min_ngram - 1, -1):
        if total_len > n and best_match_pos < 0:
            suffix_0 = tl.load(token_base + total_len - n)
            suffix_1 = tl.load(token_base + total_len - n + 1) if n >= 2 else 0
            suffix_2 = tl.load(token_base + total_len - n + 2) if n >= 3 else 0
            suffix_3 = tl.load(token_base + total_len - n + 3) if n >= 4 else 0

            scan_end = total_len - n
            num_blocks = tl.cdiv(scan_end - search_start, BLOCK_SIZE)
            # Scan backwards in vectorized blocks (latest occurrence first)
            for b in range(num_blocks - 1, -1, -1):
                if best_match_pos < 0:
                    b_start = search_start + b * BLOCK_SIZE
                    offs = b_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs < scan_end) & (offs >= search_start)

                    t0 = tl.load(token_base + offs, mask=mask, other=-999999)
                    m0 = (t0 == suffix_0) & mask

                    if n >= 2:
                        t1 = tl.load(token_base + offs + 1, mask=mask, other=-999999)
                        m0 = m0 & (t1 == suffix_1)
                    if n >= 3:
                        t2 = tl.load(token_base + offs + 2, mask=mask, other=-999999)
                        m0 = m0 & (t2 == suffix_2)
                    if n >= 4:
                        t3 = tl.load(token_base + offs + 3, mask=mask, other=-999999)
                        m0 = m0 & (t3 == suffix_3)

                    # If match found in block, select highest index
                    if tl.sum(m0.to(tl.int32)) > 0:
                        match_positions = tl.where(m0, offs, -1)
                        best_in_block = tl.max(match_positions, axis=0)
                        if best_in_block >= 0:
                            best_match_pos = best_in_block
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
    """Fast GPU-resident vectorized N-gram matcher for speculative drafting."""

    def __init__(
        self,
        max_num_reqs: int,
        max_spec_tokens: int,
        min_ngram: int = 2,
        max_ngram: int = 4,
        max_history_search: int = 4096,
        device: torch.device = torch.device("cuda"),
    ):
        if min_ngram < 2:
            raise ValueError(f"min_ngram must be >= 2, got {min_ngram}")
        if max_ngram < min_ngram:
            raise ValueError(f"max_ngram must be >= min_ngram ({min_ngram}), got {max_ngram}")
        if max_ngram > 4:
            raise ValueError(
                f"max_ngram ({max_ngram}) > 4 is not supported by the unrolled SIMD Triton kernel"
            )

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
        _vectorized_ngram_lookup_kernel[grid](
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
            BLOCK_SIZE=256,
        )

        return self.ngram_draft_tokens[:num_reqs], self.ngram_match_lens[:num_reqs]
