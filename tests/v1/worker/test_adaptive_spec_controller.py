# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from vllm.v1.worker.gpu.spec_decode.adaptive_controller import AdaptiveSpecController


def test_adaptive_controller_basic_transitions():
    controller = AdaptiveSpecController(
        max_num_reqs=4,
        n_max=7,
        n_min=2,
        alpha=0.25,
        probe_step=1.0,
        init_val=2.0,
        device=torch.device("cpu"),
        enabled=True,
    )

    req_idx = 0
    # Initial effective draft length should be max(2, min(7, round(2.0))) = 2
    l1 = controller.get_effective_draft_length(req_idx)
    assert l1 == 2
    assert controller.last_draft_len[req_idx].item() == 2

    # Step 1: Full acceptance (n_accepted = 2, n_drafted = 2)
    # num_sampled = 3 (1 bonus + 2 draft), num_rejected = 0
    controller.update_acceptance(
        num_sampled=torch.tensor([3]),
        num_rejected=torch.tensor([0]),
        idx_mapping=torch.tensor([0]),
    )
    # Full accept triggers additive probe (+1.0) -> ema becomes 3.0
    assert abs(controller.acc_ema[req_idx].item() - 3.0) < 1e-4

    # Next draft length should be 3
    l2 = controller.get_effective_draft_length(req_idx)
    assert l2 == 3

    # Step 2: Partial acceptance (n_accepted = 1, n_drafted = 3)
    # num_sampled = 2 (1 bonus + 1 draft), num_rejected = 2
    controller.update_acceptance(
        num_sampled=torch.tensor([2]),
        num_rejected=torch.tensor([2]),
        idx_mapping=torch.tensor([0]),
    )
    # Partial accept triggers EMA: (1 - 0.25) * 3.0 + 0.25 * 1.0 = 2.25 + 0.25 = 2.5
    assert abs(controller.acc_ema[req_idx].item() - 2.5) < 1e-4

    # Next draft length: round(2.5) = 3 (or 2 depending on round direction, clamped >= n_min)
    l3 = controller.get_effective_draft_length(req_idx)
    assert l3 in (2, 3)


def test_adaptive_controller_floor_and_ceiling():
    controller = AdaptiveSpecController(
        max_num_reqs=2,
        n_max=5,
        n_min=2,
        alpha=0.5,
        probe_step=2.0,
        init_val=5.0,
        device=torch.device("cpu"),
        enabled=True,
    )

    req_idx = 0
    # Probe upward beyond n_max
    controller.get_effective_draft_length(req_idx)
    controller.update_acceptance(
        num_sampled=torch.tensor([6]),
        num_rejected=torch.tensor([0]),
        idx_mapping=torch.tensor([0]),
    )
    # EMA is clamped at n_max=5.0
    assert controller.acc_ema[req_idx].item() <= 5.0
    assert controller.get_effective_draft_length(req_idx) == 5

    # Severe rejection: n_accepted = 0
    controller.update_acceptance(
        num_sampled=torch.tensor([1]),
        num_rejected=torch.tensor([5]),
        idx_mapping=torch.tensor([0]),
    )
    # 0.5 * 5.0 + 0.5 * 0 = 2.5
    assert abs(controller.acc_ema[req_idx].item() - 2.5) < 1e-4

    # Severe rejection again: 0.5 * 2.5 + 0 = 1.25
    controller.get_effective_draft_length(req_idx)
    controller.update_acceptance(
        num_sampled=torch.tensor([1]),
        num_rejected=torch.tensor([2]),
        idx_mapping=torch.tensor([0]),
    )
    assert abs(controller.acc_ema[req_idx].item() - 1.25) < 1e-4
    # But effective draft length enforces n_min=2
    assert controller.get_effective_draft_length(req_idx) == 2


def test_adaptive_controller_reset():
    controller = AdaptiveSpecController(
        max_num_reqs=2,
        n_max=7,
        n_min=1,
        init_val=3.0,
        device=torch.device("cpu"),
        enabled=True,
    )
    controller.acc_ema[0] = 6.5
    controller.last_draft_len[0] = 7
    controller.reset_request(0)
    assert controller.acc_ema[0].item() == 3.0
    assert controller.last_draft_len[0].item() == 0


def test_adaptive_controller_batch_vectorized():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    for dev in devices:
        controller = AdaptiveSpecController(
            max_num_reqs=8,
            n_max=7,
            n_min=1,
            alpha=0.25,
            probe_step=1.0,
            init_val=2.0,
            device=dev,
            enabled=True,
        )

        idx_mapping = torch.tensor([0, 1, 2, 3, -1, 99], dtype=torch.int64, device=dev)
        # Get draft lengths for batch
        eff_lens = controller.get_batch_effective_draft_lengths(idx_mapping)
        assert eff_lens.shape[0] == 6
        assert eff_lens[0].item() == 2
        assert controller.last_draft_len[0].item() == 2
        assert controller.last_draft_len[1].item() == 2

        # Step 1: Batch update with mixed outcomes
        # req 0: full accept (n_accepted=2 >= n_drafted=2) -> probe 2.0 + 1.0 = 3.0
        # req 1: partial accept (n_accepted=1 < n_drafted=2) -> decay (1-0.25)*2.0 + 0.25*1 = 1.75
        # req 2: prefill step (num_sampled=0) -> skipped, draft_len preserved
        # req 3: full accept (n_accepted=2 >= n_drafted=2) -> probe 2.0 + 1.0 = 3.0
        num_sampled = torch.tensor([3, 2, 0, 3, 3, 3], dtype=torch.int32, device=dev)
        num_rejected = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.int32, device=dev)

        controller.update_acceptance(num_sampled, num_rejected, idx_mapping)

        assert abs(controller.acc_ema[0].item() - 3.0) < 1e-4
        assert abs(controller.acc_ema[1].item() - 1.75) < 1e-4
        assert abs(controller.acc_ema[2].item() - 2.0) < 1e-4
        assert abs(controller.acc_ema[3].item() - 3.0) < 1e-4

        assert controller.last_draft_len[0].item() == 0
        assert controller.last_draft_len[1].item() == 0
        assert controller.last_draft_len[2].item() == 2  # preserved because num_sampled was 0
        assert controller.last_draft_len[3].item() == 0


if __name__ == "__main__":
    test_adaptive_controller_basic_transitions()
    test_adaptive_controller_floor_and_ceiling()
    test_adaptive_controller_reset()
    test_adaptive_controller_batch_vectorized()
    print("All AdaptiveSpecController unit tests passed!")
