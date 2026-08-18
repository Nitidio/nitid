"""Pretrained-weight transfer helpers for custom detection taxonomies."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

ENCODER_SCORE_KEYS = (
    "decoder.enc_score_head.weight",
    "decoder.enc_score_head.bias",
)
CLASS_SPECIFIC_PREFIXES = (
    "decoder.dec_score_head.",
    "decoder.denoising_class_embed.",
    "decoder.classifier.",
    "decoder.aux_head.",
)


@dataclass(frozen=True)
class ClassTransferResult:
    model: nn.Module
    config: dict
    transferred: int
    initialized: tuple[str, ...]
    mapped_proposal_scorer: tuple[str, ...]
    changed: bool


def _custom_class_transfer_state(
    pretrained_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Transfer generic features while reinitializing taxonomy-specific tensors."""
    transfer = {
        name: value
        for name, value in pretrained_state.items()
        if name in target_state
        and value.shape == target_state[name].shape
        and name not in ENCODER_SCORE_KEYS
        and not name.startswith(CLASS_SPECIFIC_PREFIXES)
    }
    mapped: list[str] = []
    for name in ENCODER_SCORE_KEYS:
        source = pretrained_state.get(name)
        target = target_state.get(name)
        if source is None or target is None:
            continue
        if source.ndim != target.ndim or source.shape[1:] != target.shape[1:]:
            raise RuntimeError(
                f"Cannot map pretrained proposal scorer {name}: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}"
            )
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise RuntimeError(
                f"Proposal scorer dimensions must be non-empty for {name}: "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}"
            )

        # D-FINE selects decoder queries by max(class logits). Repeating the
        # mean pretrained logits preserves generic object-location ranking while
        # the new decoder classification heads learn the custom taxonomy.
        generic = source.mean(dim=0, keepdim=True)
        repeats = (target.shape[0],) + (1,) * (generic.ndim - 1)
        transfer[name] = generic.repeat(repeats).to(dtype=target.dtype)
        mapped.append(name)
    return transfer, mapped


def compatible_pretrained_state(
    pretrained_state: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    """Select pretrained tensors whose names and shapes match ``model`` exactly."""
    target_state = model.state_dict()
    return {
        name: value
        for name, value in pretrained_state.items()
        if isinstance(value, torch.Tensor)
        and name in target_state
        and value.shape == target_state[name].shape
    }


def adapt_model_to_classes(
    model: nn.Module,
    config: dict,
    old_names: dict[int, str],
    new_names: dict[int, str],
) -> ClassTransferResult:
    """Return a model configured for ``new_names`` with safe pretrained transfer."""
    normalized = {int(index): str(name) for index, name in new_names.items()}
    expected_indices = list(range(len(normalized)))
    if sorted(normalized) != expected_indices:
        raise ValueError(
            "Dataset class names must use contiguous zero-based indices; "
            f"expected {expected_indices}, got {sorted(normalized)}"
        )
    if not normalized:
        raise ValueError("Dataset must define at least one class")
    current_num_classes = getattr(getattr(model, "decoder", None), "num_classes", None)
    if normalized == {
        int(index): str(name) for index, name in old_names.items()
    } and current_num_classes == len(normalized):
        return ClassTransferResult(model, config, 0, (), (), False)

    from dfine.nn.build import build_model

    updated_config = copy.deepcopy(config)
    updated_config["num_classes"] = len(normalized)
    decoder_config = updated_config.get("DFINETransformer", {})
    postprocessor_config = updated_config.get("DFINEPostProcessor")
    if isinstance(decoder_config, dict) and isinstance(postprocessor_config, dict):
        num_queries = decoder_config.get("num_queries")
        num_top_queries = postprocessor_config.get("num_top_queries")
        if isinstance(num_queries, int) and isinstance(num_top_queries, int):
            postprocessor_config["num_top_queries"] = min(
                num_top_queries, num_queries * len(normalized)
            )
    rebuilt = build_model(updated_config)
    transfer, mapped = _custom_class_transfer_state(model.state_dict(), rebuilt.state_dict())
    load_result = rebuilt.load_state_dict(transfer, strict=False)
    rebuilt.to(next(model.parameters()).device)
    return ClassTransferResult(
        model=rebuilt,
        config=updated_config,
        transferred=len(transfer),
        initialized=tuple(load_result.missing_keys),
        mapped_proposal_scorer=tuple(mapped),
        changed=True,
    )
