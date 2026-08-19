"""Stable architecture and metadata contracts for native DETRPose support.

This module intentionally contains no model implementation.  It centralizes the
upstream identity, model variants, checkpoint locations, and keypoint schemas so
the model, data, validation, and export layers share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, cast

PoseDataset = Literal["coco", "crowdpose"]
PoseSize = Literal["n", "s", "m", "l", "x"]
PoseModelName = Literal[
    "detrpose_n",
    "detrpose_s",
    "detrpose_m",
    "detrpose_l",
    "detrpose_x",
]

DETRPOSE_UPSTREAM_REPOSITORY = "https://github.com/SebastianJanampa/DETRPose"
DETRPOSE_UPSTREAM_COMMIT = "4e4a842aaa5afb3d13b40224f070bc3e8e8503f6"
DETRPOSE_UPSTREAM_LICENSE = "Apache-2.0"
DETRPOSE_RELEASE_ROOT = (
    "https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights"
)


@dataclass(frozen=True)
class PoseSchema:
    """Dataset-level keypoint semantics required by every pose component."""

    name: PoseDataset
    keypoint_names: tuple[str, ...]
    flip_idx: tuple[int, ...]
    skeleton: tuple[tuple[int, int], ...]
    oks_sigmas: tuple[float, ...]
    annotation_dim: int = 3

    def __post_init__(self) -> None:
        count = len(self.keypoint_names)
        if count < 1:
            raise ValueError("A pose schema must define at least one keypoint")
        if len(set(self.keypoint_names)) != count:
            raise ValueError("Pose keypoint names must be unique")
        if self.annotation_dim not in (2, 3):
            raise ValueError("Pose annotation_dim must be 2 or 3")
        if len(self.flip_idx) != count or sorted(self.flip_idx) != list(range(count)):
            raise ValueError("Pose flip_idx must be a permutation of all keypoint indices")
        if any(self.flip_idx[self.flip_idx[index]] != index for index in range(count)):
            raise ValueError("Pose flip_idx must be self-inverse")
        if len(self.oks_sigmas) != count or any(value <= 0 for value in self.oks_sigmas):
            raise ValueError("Pose OKS sigmas must contain one positive value per keypoint")
        if any(
            start == end or start < 0 or end < 0 or start >= count or end >= count
            for start, end in self.skeleton
        ):
            raise ValueError("Pose skeleton edges must reference two distinct keypoints")

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoint_names)


@dataclass(frozen=True)
class PoseModelSpec:
    """Checkpoint-facing architecture values for one official DETRPose size."""

    name: PoseModelName
    size: PoseSize
    backbone: str
    backbone_return_idx: tuple[int, ...]
    encoder_in_channels: tuple[int, ...]
    feature_strides: tuple[int, ...]
    hidden_dim: int
    encoder_feedforward_dim: int
    decoder_feedforward_dim: int
    encoder_expansion: float
    encoder_depth_mult: float
    decoder_layers: int
    decoder_points: int
    reg_scale: float
    num_queries: int = 60
    reg_max: int = 32
    input_size: tuple[int, int] = (640, 640)
    internal_num_classes: int = 2
    public_num_classes: int = 1

    def __post_init__(self) -> None:
        if self.name != f"detrpose_{self.size}":
            raise ValueError("Pose model name and size must agree")
        if len(self.backbone_return_idx) != len(self.encoder_in_channels):
            raise ValueError("Backbone outputs and encoder input channels must align")
        if len(self.feature_strides) != len(self.encoder_in_channels):
            raise ValueError("Feature strides and encoder input channels must align")
        positive = (
            self.hidden_dim,
            self.encoder_feedforward_dim,
            self.decoder_feedforward_dim,
            self.decoder_layers,
            self.decoder_points,
            self.num_queries,
            self.reg_max,
            *self.input_size,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Pose model dimensions must be positive")
        if self.internal_num_classes != 2 or self.public_num_classes != 1:
            raise ValueError("Official DETRPose uses two internal logits and one public class")

    @property
    def num_feature_levels(self) -> int:
        return len(self.feature_strides)


@dataclass(frozen=True)
class PoseCheckpointSpec:
    """Immutable identity for one official pretrained pose checkpoint."""

    model: PoseModelName
    dataset: PoseDataset
    filename: str
    url: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        suffix = "_crowdpose" if self.dataset == "crowdpose" else ""
        expected = f"detrpose_hgnetv2_{self.model[-1]}{suffix}.pth"
        if self.filename != expected:
            raise ValueError(f"Unexpected official checkpoint filename: {self.filename}")
        if self.url != f"{DETRPOSE_RELEASE_ROOT}/{self.filename}":
            raise ValueError("Pose checkpoint URL must reference the official DETRPose release")
        if self.sha256 is not None:
            if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
                raise ValueError(
                    "Pose checkpoint sha256 must be 64 lowercase hexadecimal characters"
                )


COCO17_SCHEMA = PoseSchema(
    name="coco",
    keypoint_names=(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    flip_idx=(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
    skeleton=(
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 11),
        (6, 12),
        (3, 5),
        (4, 6),
    ),
    oks_sigmas=(
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ),
)

CROWDPOSE14_SCHEMA = PoseSchema(
    name="crowdpose",
    keypoint_names=(
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head",
        "neck",
    ),
    flip_idx=(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13),
    skeleton=(
        (12, 13),
        (13, 0),
        (13, 1),
        (0, 2),
        (2, 4),
        (1, 3),
        (3, 5),
        (0, 6),
        (1, 7),
        (6, 7),
        (6, 8),
        (8, 10),
        (7, 9),
        (9, 11),
    ),
    oks_sigmas=(
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.079,
        0.079,
    ),
)

POSE_SCHEMAS: Mapping[PoseDataset, PoseSchema] = MappingProxyType(
    {"coco": COCO17_SCHEMA, "crowdpose": CROWDPOSE14_SCHEMA}
)


def _model_spec(
    size: PoseSize,
    *,
    backbone: str,
    return_idx: tuple[int, ...],
    in_channels: tuple[int, ...],
    strides: tuple[int, ...],
    hidden_dim: int,
    encoder_feedforward_dim: int,
    decoder_feedforward_dim: int,
    expansion: float,
    depth_mult: float,
    decoder_layers: int,
    decoder_points: int = 4,
    reg_scale: float = 4.0,
) -> PoseModelSpec:
    return PoseModelSpec(
        name=cast(PoseModelName, f"detrpose_{size}"),
        size=size,
        backbone=backbone,
        backbone_return_idx=return_idx,
        encoder_in_channels=in_channels,
        feature_strides=strides,
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=encoder_feedforward_dim,
        decoder_feedforward_dim=decoder_feedforward_dim,
        encoder_expansion=expansion,
        encoder_depth_mult=depth_mult,
        decoder_layers=decoder_layers,
        decoder_points=decoder_points,
        reg_scale=reg_scale,
    )


POSE_MODEL_SPECS: Mapping[PoseModelName, PoseModelSpec] = MappingProxyType(
    {
        "detrpose_n": _model_spec(
            "n",
            backbone="B0",
            return_idx=(2, 3),
            in_channels=(512, 1024),
            strides=(16, 32),
            hidden_dim=128,
            encoder_feedforward_dim=512,
            decoder_feedforward_dim=512,
            expansion=0.34,
            depth_mult=0.5,
            decoder_layers=3,
            decoder_points=6,
        ),
        "detrpose_s": _model_spec(
            "s",
            backbone="B0",
            return_idx=(1, 2, 3),
            in_channels=(256, 512, 1024),
            strides=(8, 16, 32),
            hidden_dim=256,
            encoder_feedforward_dim=1024,
            decoder_feedforward_dim=1024,
            expansion=0.5,
            depth_mult=0.34,
            decoder_layers=3,
        ),
        "detrpose_m": _model_spec(
            "m",
            backbone="B2",
            return_idx=(1, 2, 3),
            in_channels=(384, 768, 1536),
            strides=(8, 16, 32),
            hidden_dim=256,
            encoder_feedforward_dim=1024,
            decoder_feedforward_dim=1024,
            expansion=1.0,
            depth_mult=0.67,
            decoder_layers=4,
        ),
        "detrpose_l": _model_spec(
            "l",
            backbone="B4",
            return_idx=(1, 2, 3),
            in_channels=(512, 1024, 2048),
            strides=(8, 16, 32),
            hidden_dim=256,
            encoder_feedforward_dim=1024,
            decoder_feedforward_dim=1024,
            expansion=1.0,
            depth_mult=1.0,
            decoder_layers=6,
        ),
        "detrpose_x": _model_spec(
            "x",
            backbone="B5",
            return_idx=(1, 2, 3),
            in_channels=(512, 1024, 2048),
            strides=(8, 16, 32),
            hidden_dim=384,
            encoder_feedforward_dim=2048,
            decoder_feedforward_dim=1024,
            expansion=1.0,
            depth_mult=1.0,
            decoder_layers=6,
            reg_scale=8.0,
        ),
    }
)

POSE_MODEL_NAMES: tuple[PoseModelName, ...] = tuple(POSE_MODEL_SPECS)


def normalize_pose_model(model: str) -> PoseModelName:
    """Return a canonical DETRPose model name without ambiguous size-only aliases."""
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}")
    normalized = model.lower().strip().replace("-", "_")
    if normalized.startswith("detrpose") and not normalized.startswith("detrpose_"):
        normalized = normalized.replace("detrpose", "detrpose_", 1)
    if normalized not in POSE_MODEL_SPECS:
        choices = ", ".join(POSE_MODEL_NAMES)
        raise ValueError(f"Unsupported DETRPose model {model!r}. Choose: {choices}")
    return cast(PoseModelName, normalized)


def validate_pose_model_task(model: str, task: str) -> PoseModelName:
    """Validate the public model/task pairing reserved for DETRPose."""
    if not isinstance(task, str):
        raise TypeError(f"task must be a string, got {type(task).__name__}")
    normalized_task = task.lower().strip().replace("-", "_")
    if normalized_task != "pose":
        raise ValueError("DETRPose architectures support task='pose' only")
    return normalize_pose_model(model)


def get_pose_model_spec(model: str) -> PoseModelSpec:
    return POSE_MODEL_SPECS[normalize_pose_model(model)]


def get_pose_schema(dataset: str) -> PoseSchema:
    if not isinstance(dataset, str):
        raise TypeError(f"dataset must be a string, got {type(dataset).__name__}")
    normalized = dataset.lower().strip().replace("-", "")
    aliases: Mapping[str, PoseDataset] = {
        "coco": "coco",
        "coco17": "coco",
        "crowdpose": "crowdpose",
        "crowdpose14": "crowdpose",
    }
    try:
        return POSE_SCHEMAS[aliases[normalized]]
    except KeyError as error:
        raise ValueError("Unsupported pose schema. Choose: coco, crowdpose") from error


def get_pose_checkpoint(model: str, dataset: str = "coco") -> PoseCheckpointSpec:
    model_name = normalize_pose_model(model)
    schema = get_pose_schema(dataset)
    suffix = "_crowdpose" if schema.name == "crowdpose" else ""
    filename = f"detrpose_hgnetv2_{model_name[-1]}{suffix}.pth"
    verified_hashes: Mapping[tuple[PoseModelName, PoseDataset], str] = {
        ("detrpose_n", "coco"): ("802014f3929b67d0ea7de068e66f11fc836310ecaf814dd3845b703d16256fac")
    }
    return PoseCheckpointSpec(
        model=model_name,
        dataset=schema.name,
        filename=filename,
        url=f"{DETRPOSE_RELEASE_ROOT}/{filename}",
        sha256=verified_hashes.get((model_name, schema.name)),
    )
