from __future__ import annotations

import pytest
import torch


def test_rio_obb_configs_cover_nitid_sizes() -> None:
    from dfine.nn.rio import RIO_OBB_MODEL_SIZES, make_rio_obb_config

    assert RIO_OBB_MODEL_SIZES == ("n", "s", "m", "l", "x")
    for size in RIO_OBB_MODEL_SIZES:
        cfg = make_rio_obb_config(f"nitid1{size}", num_classes=3, image_size=(128, 128))
        assert cfg["task"] == "obb"
        assert cfg["num_classes"] == 3
        assert cfg["eval_spatial_size"] == [128, 128]
        assert cfg["RioOBB"]["decoder"] == "RTDETRTransformerv2OBB"


def test_native_rio_obb_model_construction_for_all_sizes() -> None:
    from dfine.nn.native_build import build_native_model
    from dfine.nn.rio import RioOBBModel

    for size in ("n", "s", "m", "l", "x"):
        model = build_native_model(
            f"nitid1{size}", num_classes=3, task="obb", image_size=(256, 256)
        )
        assert isinstance(model, RioOBBModel)
        assert model.decoder.num_classes == 3
        assert model.decoder.num_queries == 300


def test_native_rio_obb_forward_smoke() -> None:
    from dfine.nn.native_build import build_native_model

    model = build_native_model("nitid1n", num_classes=3, task="obb", image_size=(256, 256)).eval()

    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, 256, 256))

    assert outputs["pred_logits"].shape == (1, 300, 3)
    assert outputs["pred_boxes"].shape == (1, 300, 5)
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()


def test_native_rio_obb_criterion_and_postprocessor_construction() -> None:
    from dfine.nn.build import build_postprocessor
    from dfine.nn.native_build import build_native_criterion
    from dfine.nn.rio import PostProcessorOBB, RTv4OBBCriterion, make_rio_obb_config

    criterion = build_native_criterion("nitid1n", num_classes=3, task="obb")
    postprocessor = build_postprocessor(make_rio_obb_config("nitid1n", num_classes=3))

    assert isinstance(criterion, RTv4OBBCriterion)
    assert isinstance(postprocessor, PostProcessorOBB)


def test_rio_obb_postprocessor_scales_width_height_in_predictor_order() -> None:
    from dfine.nn.rio import PostProcessorOBB

    postprocessor = PostProcessorOBB(num_classes=2, num_top_queries=1)
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -8.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.25, 0.25, 0.5, 0.25]]]),
    }

    result = postprocessor(outputs, torch.tensor([[200.0, 100.0]]))[0]

    torch.testing.assert_close(
        result["boxes"][0], torch.tensor([100.0, 25.0, 50.0, 50.0, torch.pi / 4])
    )
    assert int(result["labels"][0]) == 0
    assert float(result["scores"][0]) == pytest.approx(float(torch.sigmoid(torch.tensor(8.0))))


def test_native_rio_obb_rejects_invalid_size() -> None:
    from dfine.nn.native_build import build_native_model

    with pytest.raises(ValueError, match="Unsupported RiO-DETR OBB model"):
        build_native_model("nitid1tiny", num_classes=3, task="obb")
