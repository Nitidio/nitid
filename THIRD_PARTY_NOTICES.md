# Third-party notices

## D-FINE and D-FINE-seg model core

The native model implementation under `dfine/nn/architecture`, the model-size
configuration in `dfine/nn/configs.py`, the detection postprocessor in
`dfine/nn/postprocessor.py`, and the losses under `dfine/nn/losses` contain
code derived from:

- D-FINE, Copyright (c) 2024 The D-FINE Authors.
  Source: https://github.com/Peterande/D-FINE
- D-FINE-seg 0.2.0, Copyright (c) 2026 The D-FINE-seg Authors.
  Source: https://github.com/ArgoHA/D-FINE-seg
- RT-DETR, Copyright (c) 2023 lyuwenyu.
  Source: https://github.com/lyuwenyu/RT-DETR

These components are provided under the Apache License 2.0. The repository's
root `LICENSE` file contains the license text. Files retain the applicable
upstream copyright and modification notices.

The imported code has been modified for nitid, including package-relative
imports, removal of application-framework dependencies, native task/model
validation, and composition through nitid's model builder.

## DETRPose pose-estimation architecture

The DETRPose architecture contract and the native pose implementation derived
from it are based on:

- DETRPose, Copyright (c) 2025 The DETRPose Authors.
  Source: https://github.com/SebastianJanampa/DETRPose
  Audited commit: `4e4a842aaa5afb3d13b40224f070bc3e8e8503f6`

DETRPose is provided under the Apache License 2.0. The repository's root
`LICENSE` file contains the license text. The nitid implementation uses its own
package, task, device, data, and training abstractions; it does not require the
upstream repository at runtime.

## RiO-DETR / RT-DETRv2-OBB

The oriented detection components under `dfine/nn/rio/` and the DOTA parsing
and target conventions in `dfine/utils/data.py` are adapted from the released
RiO-DETR / RT-DETRv2-OBB framework.

- Source: https://github.com/RicePasteM/RiO-DETR
- Audited revision: `58dd5300a15296e4db14be5dcc89ebf5235fc064`
- License: Apache-2.0 (`LICENSE` at that revision).

The released framework retains copyright notices from lyuwenyu, Facebook,
the D-FINE Authors, the DEIM Authors, and the RT-DETRv4 Authors in its derived
files. Those notices are preserved in nitid. Some RiO-specific files have no
separate copyright header; no year or copyright-holder statement has been
invented for those files.

Nitid modifications include package-relative imports, removal of upstream
registries, native model/config composition, dataset handling, and integration
with Nitid's prediction, training, validation, and export interfaces. The
reference repository is not a runtime dependency. This attribution covers the
released framework, not the unreleased paper-specific improvements described
in its README. See `docs/adr/0003-native-rio-detr-obb-integration.md` for the
source extraction map.

## DETR

Bounding-box operations in `dfine/nn/rio/box_ops.py`, matching code in
`dfine/nn/losses/matcher.py` and `dfine/nn/rio/obb_matcher.py`, and the frozen
batch-normalization implementation in `dfine/nn/architecture/common.py`
contain code derived from DETR, including through RT-DETR and D-FINE.

- Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
- Source: https://github.com/facebookresearch/detr
- Audited revision: `29901c51d7fe8712168b8d0d64351170bc0f83e0`
- License: Apache-2.0 (`LICENSE` at that revision).

Nitid modifications include package integration, task-specific matching and
oriented-box handling. Existing upstream notices are retained in the source.

## PaddleDetection HGNetv2

`dfine/nn/architecture/backbone.py` contains an adaptation of PaddleDetection's
HGNetv2 backbone through D-FINE's PyTorch implementation.

- Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
- Source: https://github.com/PaddlePaddle/PaddleDetection
- Upstream file: `ppdet/modeling/backbones/hgnet_v2.py`
- Audited revision: `b25522a0f4bde8c80603f3ba5e3472059972e3b5`
- License: Apache-2.0 (`LICENSE` and the upstream file header).

The PaddlePaddle and D-FINE notices are retained. Nitid modifications include
native package integration and model construction.

## DEIM and RT-DETRv4 code retained through RiO-DETR

The OBB decoder, criterion, and utilities in `dfine/nn/rio/` retain upstream
code and notices from the following projects. This attribution does not imply
that Nitid offers their complete model families.

- DEIM, Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
  The upstream license also states: Copyright (C) INTELLINDUST INFORMATION
  TECHNOLOGY (SHENZHEN) CO., LTD. and all its affiliates.
  Source: https://github.com/ShihuaHuang95/DEIM
  Audited revision: `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`
- RT-DETRv4, Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
  Source: https://github.com/RT-DETRs/RT-DETRv4
  Audited revision: `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`

Both upstream `LICENSE` files specify Apache-2.0. Nitid modifications include
package integration and OBB task adaptation, as noted in the source files.

The audited revisions above identify the upstream sources inspected for this
notice update; they are not claims about the exact commits originally imported.
The repository's root `LICENSE` supplies the Apache-2.0 license text and is
included with this notice in the wheel.
