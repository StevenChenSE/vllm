# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.v1.worker.gpu.spec_decode.ngram_lookup import NgramLookupModule


def test_ngram_lookup_exact_matches():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA/ROCm not available, skipping test")
        return

    module = NgramLookupModule(
        max_num_reqs=4,
        max_spec_tokens=5,
        min_ngram=2,
        max_ngram=3,
        max_history_search=1024,
        device=device,
    )

    # Sequence 0: "The quick brown fox jumps over the lazy dog. The quick brown"
    # Tokens: [10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30]
    # Suffix 3-gram: [10, 20, 30]
    # Match at index 0: followed by [40, 50, 60, 70, 80]
    seq0 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20, 30]

    # Sequence 1: No match (all unique)
    seq1 = [1, 2, 3, 4, 5, 6, 7]

    all_tokens = torch.zeros((4, 64), dtype=torch.int32, device=device)
    all_tokens[0, : len(seq0)] = torch.tensor(seq0, dtype=torch.int32, device=device)
    all_tokens[1, : len(seq1)] = torch.tensor(seq1, dtype=torch.int32, device=device)

    total_lens = torch.tensor([len(seq0), len(seq1), 0, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0, 1, -1], dtype=torch.int64, device=device)

    drafts, match_lens = module.lookup(
        all_token_ids=all_tokens,
        total_lens=total_lens,
        idx_mapping=idx_mapping,
        num_reqs=3,
    )

    print("Seq 0 match len:", match_lens[0].item(), "drafts:", drafts[0].tolist())
    print("Seq 1 match len:", match_lens[1].item(), "drafts:", drafts[1].tolist())
    print("Seq 2 match len:", match_lens[2].item(), "drafts:", drafts[2].tolist())

    assert match_lens[0].item() == 5
    assert drafts[0].tolist() == [40, 50, 60, 70, 80]

    assert match_lens[1].item() == 0
    assert drafts[1].tolist() == [-1, -1, -1, -1, -1]

    assert match_lens[2].item() == 0
    assert drafts[2].tolist() == [-1, -1, -1, -1, -1]
    print("test_ngram_lookup_exact_matches passed!")


if __name__ == "__main__":
    test_ngram_lookup_exact_matches()
