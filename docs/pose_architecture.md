# DETRPose architecture contract

nitid treats DETRPose as a dedicated pose-estimation family. It shares useful D-FINE foundations, but
its official weights depend on a pose decoder and numerical details that are not interchangeable with a
D-FINE detection checkpoint.

The reserved public API is:

```python
DFINE("detrpose_n", task="pose")
```

This API is operational for native inference with wrapped DETRPose checkpoints and official DETRPose
checkpoint conversion. Pose training and validation use COCO keypoint annotations and keep the public
task surface constrained to single-class person pose estimation.

## Model family

The immutable model catalog lives in `dfine.pose_contract`. Its values mirror the audited official
configuration rather than inferring pose settings from similarly named D-FINE models.

| Model | Backbone | Encoder width | Decoder layers | Queries | Input contract |
|---|---:|---:|---:|---:|---:|
| `detrpose_n` | HGNetV2-B0 | 128 | 3 | 60 | 640 × 640 |
| `detrpose_s` | HGNetV2-B0 | 256 | 3 | 60 | 640 × 640 |
| `detrpose_m` | HGNetV2-B2 | 256 | 4 | 60 | 640 × 640 |
| `detrpose_l` | HGNetV2-B4 | 256 | 6 | 60 | 640 × 640 |
| `detrpose_x` | HGNetV2-B5 | 384 | 6 | 60 | 640 × 640 |

Size-only names such as `n` and D-FINE names such as `dfine_n` are deliberately rejected for pose.
This prevents a checkpoint from silently selecting the wrong architecture family.

## Dataset schemas

Two keypoint schemas are defined:

- **COCO-17**: 17 keypoints in COCO order.
- **CrowdPose-14**: 14 keypoints in CrowdPose order.

Each schema owns its keypoint names, horizontal-flip permutation, visualization skeleton, annotation
dimension, and OKS sigmas. These values are validated together so augmentation, losses, metrics, and
rendering cannot develop independent keypoint conventions.

## Checkpoint and output semantics

Official checkpoint URLs are tied to the official DETRPose release. The audited
`detrpose_hgnetv2_n.pth` COCO checkpoint has SHA-256:

```text
802014f3929b67d0ea7de068e66f11fc836310ecaf814dd3845b703d16256fac
```

Other official checkpoint locations are cataloged, but each checksum must be independently verified
before the checkpoint is admitted to automatic downloads.

The official decoder emits 60 queries, two internal classification logits per query, and flattened
normalized `(x, y)` keypoint coordinates. The public contract exposes one `person` class. It does not
claim native bounding boxes or learned per-keypoint confidence values because the architecture does not
produce them.

## Compatibility gates

A native implementation is compatible only when all of the following hold:

1. Official weights load strictly into the intended model variant.
2. Raw output shapes and selected values match the pinned upstream golden fixture within its declared
   tolerance.
3. Postprocessed coordinates remain correct through resize and letterbox reversal.
4. Existing detection, instance-segmentation, and semantic-segmentation outputs are unchanged.
5. CPU and supported accelerators use device-neutral code paths.

The upstream provenance, architectural decision, and accepted integration boundaries are recorded in
[ADR-0002](adr/0002-native-detrpose-integration.md).
