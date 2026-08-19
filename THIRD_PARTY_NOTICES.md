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
