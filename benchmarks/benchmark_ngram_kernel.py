# SPDX-License-Identifier: Apache-2.0
import time
import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.spec_decode.ngram_lookup import NgramLookupModule

# Vectorized block-parallel Triton kernel
@triton.jit
def _ngram_lookup_block_kernel(
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

    # Test each n-gram size from max down to min
    for n in range(max_ngram, min_ngram - 1, -1):
        if total_len > n and best_match_pos < 0:
            suffix_0 = tl.load(token_base + total_len - n)
            suffix_1 = tl.load(token_base + total_len - n + 1) if n >= 2 else 0
            suffix_2 = tl.load(token_base + total_len - n + 2) if n >= 3 else 0
            suffix_3 = tl.load(token_base + total_len - n + 3) if n >= 4 else 0

            # Scan in chunks of BLOCK_SIZE backwards
            scan_end = total_len - n
            # Number of candidate start positions
            num_candidates = scan_end - search_start
            if num_candidates > 0:
                # We can step backwards in blocks
                cur_pos = scan_end - 1
                while cur_pos >= search_start and best_match_pos < 0:
                    # Check pos
                    m0 = tl.load(token_base + cur_pos) == suffix_0
                    m1 = (tl.load(token_base + cur_pos + 1) == suffix_1) if n >= 2 else True
                    m2 = (tl.load(token_base + cur_pos + 2) == suffix_2) if n >= 3 else True
                    m3 = (tl.load(token_base + cur_pos + 3) == suffix_3) if n >= 4 else True

                    if m0 and m1 and m2 and m3:
                        best_match_pos = cur_pos
                        best_match_n = n
                    cur_pos -= 1

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


def bench_ngram():
    device = torch.device("cuda:0")
    module = NgramLookupModule(
        max_num_reqs=8,
        max_spec_tokens=7,
        min_ngram=2,
        max_ngram=4,
        max_history_search=4096,
        device=device,
    )

    batch_size = 8
    seq_len = 4096
    all_tokens = torch.randint(0, 50000, (batch_size, seq_len), dtype=torch.int32, device=device)
    # Insert match in every sequence at pos 1000
    all_tokens[:, 1000:1003] = all_tokens[:, seq_len-3:seq_len]
    total_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(batch_size, dtype=torch.int64, device=device)

    # Warmup
    for _ in range(10):
        module.lookup(all_tokens, total_lens, idx_mapping, batch_size)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        module.lookup(all_tokens, total_lens, idx_mapping, batch_size)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    us_per_call = (t1 - t0) / iters * 1e6
    print(f"Ngram lookup latency for batch_size={batch_size}, seq_len={seq_len}: {us_per_call:.2f} μs")


if __name__ == "__main__":
    bench_ngram()
