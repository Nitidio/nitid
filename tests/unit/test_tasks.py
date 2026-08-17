"""Unit tests for public task normalization and capability contracts."""

import pytest

from dfine.tasks import SUPPORTED_TASKS, get_task_contract, normalize_task


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("detect", "detect"),
        (" Detection ", "detect"),
        ("segment", "segment"),
        ("semantic", "semantic"),
        ("SEM_SEG", "semantic"),
        ("sem-seg", "semantic"),
    ],
)
def test_normalize_task_accepts_canonical_names_and_documented_aliases(value, expected):
    assert normalize_task(value) == expected


def test_normalize_task_rejects_unknown_and_non_string_values():
    with pytest.raises(ValueError, match="detect, segment, semantic"):
        normalize_task("panoptic")
    with pytest.raises(TypeError, match="task must be a string"):
        normalize_task(1)  # type: ignore[arg-type]


def test_semantic_task_contract_is_dense_and_uses_instance_pretraining():
    contract = get_task_contract("sem_seg")

    assert SUPPORTED_TASKS == ("detect", "segment", "semantic")
    assert contract.name == "semantic"
    assert contract.result_fields == ("semantic_mask",)
    assert contract.instance_level is False
    assert contract.pretrained_source_task == "segment"
