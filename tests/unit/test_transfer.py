"""Tests for pretrained custom-taxonomy transfer."""

import torch

from dfine.nn.transfer import _custom_class_transfer_state


def test_custom_class_transfer_repeats_generic_proposal_scorer():
    pretrained = {
        "backbone.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "decoder.enc_score_head.weight": torch.tensor(
            [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]]
        ),
        "decoder.enc_score_head.bias": torch.tensor([-5.0, -4.0, -3.0]),
        "decoder.dec_score_head.0.weight": torch.ones(3, 3),
    }
    target = {
        "backbone.weight": torch.empty(2, 3),
        "decoder.enc_score_head.weight": torch.empty(2, 3),
        "decoder.enc_score_head.bias": torch.empty(2),
        "decoder.dec_score_head.0.weight": torch.empty(2, 3),
    }

    transfer, mapped = _custom_class_transfer_state(pretrained, target)

    assert transfer["backbone.weight"] is pretrained["backbone.weight"]
    assert torch.equal(
        transfer["decoder.enc_score_head.weight"],
        torch.tensor([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]),
    )
    assert torch.equal(transfer["decoder.enc_score_head.bias"], torch.tensor([-4.0, -4.0]))
    assert "decoder.dec_score_head.0.weight" not in transfer
    assert mapped == [
        "decoder.enc_score_head.weight",
        "decoder.enc_score_head.bias",
    ]


def test_custom_class_transfer_reinitializes_same_shape_taxonomy_heads():
    pretrained = {
        "decoder.enc_score_head.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "decoder.enc_score_head.bias": torch.tensor([-2.0, -4.0]),
        "decoder.dec_score_head.0.weight": torch.ones(2, 3),
        "decoder.denoising_class_embed.weight": torch.ones(3, 3),
    }
    target = {name: torch.empty_like(value) for name, value in pretrained.items()}

    transfer, _ = _custom_class_transfer_state(pretrained, target)

    assert torch.equal(
        transfer["decoder.enc_score_head.weight"],
        torch.tensor([[1.5, 2.5, 3.5], [1.5, 2.5, 3.5]]),
    )
    assert "decoder.dec_score_head.0.weight" not in transfer
    assert "decoder.denoising_class_embed.weight" not in transfer
