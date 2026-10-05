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

## DETR

Matching code in `dfine/nn/losses/matcher.py` and the frozen
batch-normalization implementation in `dfine/nn/architecture/common.py`
contain code derived from DETR, including through RT-DETR and D-FINE.

- Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
- Source: https://github.com/facebookresearch/detr
- Audited revision: `29901c51d7fe8712168b8d0d64351170bc0f83e0`
- License: Apache-2.0 (`LICENSE` at that revision).

Nitid modifications include package integration and task-specific matching.
Existing upstream notices are retained in the source.

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

## DEIM

The following follow the formulation and code of DEIM:

- the matchability-aware classification loss (`loss_labels_mal`) in
  `dfine/nn/losses/criterion.py`;
- the flat-cosine learning-rate schedule (`FlatCosineLRScheduler`) in
  `dfine/trainer.py`;
- the batch-level MixUp collate (`DetectionBatchCollate`) in
  `dfine/utils/data.py`;
- the `recipe="deim"` training defaults in `dfine/training_recipes.py`.

This attribution does not imply that Nitid offers the complete DEIM model
family.

- DEIM, Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
  The upstream license also states: Copyright (C) INTELLINDUST INFORMATION
  TECHNOLOGY (SHENZHEN) CO., LTD. and all its affiliates.
  Source: https://github.com/ShihuaHuang95/DEIM
  Audited revision: `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`
- License: Apache-2.0 (`LICENSE` at that revision).

The audited revisions above identify the upstream sources inspected for this
notice update; they are not claims about the exact commits originally imported.
The repository's root `LICENSE` supplies the Apache-2.0 license text and is
included with this notice in the wheel.
