# SPDX-License-Identifier: Apache-2.0
import time
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

    req_idx = tl.load(idx_mapping_ptr + pid)
    if req_idx < 0:
        return

    total_len = tl.load(total_lens_ptr + req_idx)
    if total_len <= min_ngram:
        tl.store(ngram_match_len_ptr + pid, 0)
        return

    token_base = all_token_ids_ptr + req_idx * all_token_ids_stride

    search_start = 0
    if total_len > max_history_search:
        search_start = total_len - max_history_search

    best_match_pos = -1
    best_match_n = 0

    # Test N-gram from max down to min (e.g. 3, 2)
    for n in range(max_ngram, min_ngram - 1, -1):
        if total_len > n and best_match_pos < 0:
            suffix_0 = tl.load(token_base + total_len - n)
            suffix_1 = tl.load(token_base + total_len - n + 1) if n >= 2 else 0
            suffix_2 = tl.load(token_base + total_len - n + 2) if n >= 3 else 0
            suffix_3 = tl.load(token_base + total_len - n + 3) if n >= 4 else 0

            scan_end = total_len - n
            # Vectorized block loop: process BLOCK_SIZE candidate positions in parallel!
            # Loop backwards in blocks from scan_end down to search_start
            num_blocks = tl.cdiv(scan_end - search_start, BLOCK_SIZE)
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

                    # Check if any position in the block matched
                    if tl.sum(m0.to(tl.int32)) > 0:
                        # Find the highest index (most recent) matching position
                        # Replace non-matches with -1
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


def bench_vectorized():
    device = torch.device("cuda:0")
    batch_size = 8
    max_spec_tokens = 7
    seq_len = 4096

    ngram_draft_tokens = torch.full((batch_size, max_spec_tokens), -1, dtype=torch.int64, device=device)
    ngram_match_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    all_tokens = torch.randint(0, 50000, (batch_size, seq_len), dtype=torch.int32, device=device)
    all_tokens[:, 1000:1003] = all_tokens[:, seq_len-3:seq_len]
    total_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(batch_size, dtype=torch.int64, device=device)

    def run_kernel():
        grid = (batch_size,)
        _vectorized_ngram_lookup_kernel[grid](
            all_tokens,
            all_tokens.stride(0),
            total_lens,
            idx_mapping,
            ngram_draft_tokens,
            ngram_draft_tokens.stride(0),
            ngram_match_lens,
            num_reqs=batch_size,
            max_spec_tokens=max_spec_tokens,
            min_ngram=2,
            max_ngram=4,
            max_history_search=4096,
            BLOCK_SIZE=256,
        )

    # Warmup
    for _ in range(10):
        run_kernel()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        run_kernel()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    us_per_call = (t1 - t0) / iters * 1e6
    print(f"Vectorized Ngram lookup latency for batch_size={batch_size}, seq_len={seq_len}: {us_per_call:.2f} μs")
    print(f"Match lens: {ngram_match_lens.tolist()}")
    print(f"Draft tokens (row 0): {ngram_draft_tokens[0].tolist()}")

if __name__ == "__main__":
    bench_vectorized()
