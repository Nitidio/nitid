"""Tests for pretrained custom-taxonomy transfer."""

import torch

from dfine.nn.transfer import _custom_class_transfer_state, adapt_model_to_classes


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


def test_semantic_taxonomy_transfer_keeps_features_and_reinitializes_classifiers():
    from dfine.nn.build import build_model
    from dfine.nn.configs import make_model_config

    config = make_model_config("dfine_n", task="semantic", num_classes=3, image_size=(64, 64))
    model = build_model(config)
    result = adapt_model_to_classes(
        model,
        config,
        {0: "old_a", 1: "old_b", 2: "old_c"},
        {0: "new_a", 1: "new_b", 2: "new_c"},
    )

    assert result.changed
    assert any(name.startswith("decoder.classifier.") for name in result.initialized)
    assert any(name.startswith("decoder.aux_head.") for name in result.initialized)
    assert torch.equal(
        result.model.state_dict()["decoder.mask_decoder.lateral.0.weight"],
        model.state_dict()["decoder.mask_decoder.lateral.0.weight"],
    )
