# ADR-0003: Native RiO-DETR OBB integration

- **Status**: Superseded (previously Proposed) —
  Pose/OBB support was removed from the library on 2026-09-25 (#193/#194). The implementation is
  preserved in the `archive/pose-obb-gstreamer` tag.
- **Date**: 2026-09-07
- **Deciders**: nitid maintainers
- **Related**: [ADR-0002](0002-native-detrpose-integration.md),
  [RiO-DETR](https://github.com/RicePasteM/RiO-DETR)

## Context

nitid is expanding from D-FINE-family detection, instance segmentation, semantic segmentation, and
DETRPose into oriented object detection. The public API is also moving toward a model-family-neutral
constructor:

```python
from nitid import NITID

model = NITID("model1s", task="obb")
```

The local `RiO-DETR-master` checkout contains the OBB implementation chosen for this integration. The
released code is a clean RT-DETRv2-OBB framework, not the complete paper implementation. Its README
states that paper-specific RiO-DETR improvements were not released because of institutional IP
restrictions, while the released framework includes baseline RT-DETR-OBB code and pretrained weights.
This means nitid should advertise the integration as RiO-DETR/RT-DETRv2-OBB-backed oriented detection,
not as a full reproduction of unreleased paper internals.

The OBB path is structurally separate from D-FINE segmentation and DETRPose. It includes its own
decoder, denoising logic, criterion, matcher, postprocessor, DOTA-style datasets, OBB transforms, and
evaluation code. The integration must therefore be native and registry-driven rather than a small
conditional branch inside existing D-FINE detection code.

## Source extraction map

The audit identified these RiO-DETR source areas as integration candidates:

| Area | RiO-DETR source | Purpose in nitid |
|---|---|---|
| OBB decoder | `engine/rtv4/rtdetrv2_obb_decoder.py` | Build RT-DETRv2-OBB heads for `task="obb"` |
| OBB denoising | `engine/rtv4/denoising_obb.py` | Training-time denoising groups |
| OBB criterion | `engine/rtv4/rtv4_obb_criterion.py` | `loss_focal`, `loss_l1`, `loss_kld` |
| OBB matcher | `engine/rtv4/obb_matcher.py` | Hungarian matching with class, KLD/GD, and optional Hausdorff costs |
| OBB box math | `engine/rtv4/utils_obb.py` and `engine/rtv4/box_ops.py` | KLD/GD loss and rotated-box utilities |
| OBB postprocessor | `engine/rtv4/postprocessor.py::PostProcessorOBB` | Convert logits and normalized OBB predictions to deployment outputs |
| OBB transforms | `engine/data/transforms/transforms_obb.py` and `mosaic_obb.py` | Resize, flip, crop, mosaic, sanitize, and convert OBB annotations |
| DOTA dataset | `engine/data/dataset/dota_dataset.py` | Parse DOTA text annotations |
| DIOR-R/FAIR1M/HRSC adapters | `engine/data/dataset/dior_dataset.py`, `fair1m_dataset.py`, `hrsc_dataset.py` | Optional compatibility targets after DOTA path is stable |
| Evaluator | `engine/data/dataset/dota_evaluator.py` | DOTA-style OBB mAP and submission artifacts |
| Visualization | `engine/misc/vis_obb.py` | Reference plotting behavior for rotated boxes |
| Export references | `tools/deployment/export_onnx.py`, `tools/inference/onnx_inf.py` | Shape/output contract checks for deployment |

The integration should not import from `RiO-DETR-master` at runtime and should not package the
reference checkout. Adapted code must live inside the installable nitid package with attribution where
required. The repository-root `THIRD_PARTY_NOTICES.md` records the released
framework attribution, inherited upstream notices, and audited source revisions.

## OBB representation contract

RiO-DETR uses a five-value oriented box internally:

```text
cx, cy, w, h, angle_norm
```

where `angle_norm` is normalized over `[0, 1]` and maps to `[0, 180]` degrees. The released
postprocessor converts normalized predictions to:

```text
cx, cy, w, h, angle_radians
```

by multiplying the angle channel by pi. It returns:

```text
labels, boxes, scores
```

in deploy mode, where `boxes` has shape `[B, N, 5]`.

nitid should keep the RiO-compatible representation internally for checkpoint compatibility, but expose
a more ergonomic public result object:

```python
result.obb.xywhr      # cx, cy, w, h, angle in radians
result.obb.xyxyxyxy   # polygon corners for drawing/export consumers
result.obb.conf
result.obb.cls
```

For ONNX export, the first supported contract should mirror RiO-DETR deploy mode:

```text
labels, boxes, scores
```

with `boxes = [cx, cy, w, h, angle_radians]`. Polygon output can be added later if deployment consumers
need an angle-free format.

## Model and weight metadata

The checked-in model zoo manifests under `RiO-DETR-master/model_zoo/rtdetrv2_obb/` define released
weights hosted on Hugging Face at `RicePasteM/RT-DETR-OBB`.

Supported release families found in the manifests:

| Weights key | Variants | Notes |
|---|---:|---|
| `diorr` | `n`, `s`, `m`, `l`, `x` | DIOR-R pretrained OBB weights |
| `dota_1_ss` | `n`, `s`, `m`, `l`, `x` | DOTA-v1.0 single-scale weights |
| `dota_1_ms` | `m`, `x` | DOTA-v1.0 multi-scale weights |

Suggested nitid aliases:

| User value | Registry key |
|---|---|
| `default` | `dota_1_ss` |
| `dota`, `dota_ss` | `dota_1_ss` |
| `dota_ms` | `dota_1_ms` |
| `diorr` | `diorr` |

`NITID("model1s", task="obb", weights="default")` should resolve to the RT-DETRv2-OBB-S DOTA-v1.0
single-scale checkpoint unless later benchmarks justify a different default.

## Dataset contract

The first supported OBB dataset format should be DOTA-style because it is the native path in
RiO-DETR:

```text
dataset/
├── train/
│   ├── images/
│   └── annfiles/
└── val/
    ├── images/
    └── annfiles/
```

Each annotation file contains one object per line:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

nitid dataset YAML should remain explicit and task-aware:

```yaml
task: obb
path: datasets/dota8
train: train/images
val: val/images
train_ann: train/annfiles
val_ann: val/annfiles
names:
  0: plane
  1: ship
```

YOLO OBB labels can be added after DOTA parsing, training, and validation are stable.

## Options considered

1. **Import `RiO-DETR-master` directly.** This is fastest initially, but leaks another registry,
   training framework, config loader, dependency surface, and path assumptions into nitid.
2. **Vendor the full RiO-DETR tree.** This preserves upstream structure but makes nitid look like a
   wrapper around multiple application repositories and complicates packaging, testing, and docs.
3. **Native port of only required OBB components.** This requires more careful extraction but keeps
   nitid installable, coherent, task-aware, and compatible with its existing training/inference/export
   conventions.
4. **Reimplement OBB support from scratch.** This maximizes control but loses compatibility with
   released RiO-DETR checkpoints and repeats substantial tested upstream work.

## Decision

Choose option 3. RiO-DETR OBB support will be implemented natively inside nitid. The reference checkout
is development-only and must not be imported or required at runtime.

The integration will proceed in this order:

1. add the model-family-neutral `NITID` registry and model-name parser;
2. port the minimal RT-DETRv2-OBB architecture, postprocessor, criterion, matcher, and OBB utilities;
3. register cleaned OBB configs for `model1{n,s,m,l,x}`;
4. add task-aware pretrained weight resolution from the RiO-DETR manifests;
5. add native `Results.obb` representation and plotting;
6. enable prediction and export;
7. add DOTA-style dataset parsing;
8. add OBB training and validation;
9. update docs, examples, packaging checks, and release tests.

## Consequences

nitid becomes responsible for maintaining the adapted OBB path, including checkpoint compatibility,
device support, tests, docs, and attribution. This is more work than direct importing, but it keeps the
library self-contained and avoids another long-lived external runtime dependency.

The first OBB release should be conservative: DOTA-style data, RiO-compatible `[cx, cy, w, h,
angle_radians]` prediction/export boxes, and the pretrained variants verified from the provided
manifests. DIOR-R, FAIR1M, HRSC2016, YOLO OBB labels, polygon export, and DOTA online submission
workflows should be added only after the core OBB path is tested.

Because the released RiO-DETR repository explicitly excludes some paper-specific improvements, nitid
must avoid overclaiming paper-level RiO-DETR reproduction. The supported feature should be described as
native oriented object detection using the released RiO-DETR/RT-DETRv2-OBB implementation and weights.
