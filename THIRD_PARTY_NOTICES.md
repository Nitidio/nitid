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

## Pretrained weights

nitid does not redistribute pretrained weights. The official checkpoints are
downloaded on first use from the sources their authors publish them in, and
keep the licence given there:

- D-FINE detection checkpoints (`nitid1n`-`nitid1x`, `task="detect"`):
  https://github.com/Peterande/storage/releases/tag/dfinev1.0, published with
  https://github.com/Peterande/D-FINE.
  License: Apache-2.0.
- D-FINE-seg instance-segmentation checkpoints (`task="segment"`):
  https://huggingface.co/ArgoSA/D-FINE-seg.
  License: Apache-2.0 (model card).

The checkpoints are trained on public datasets (COCO, and Objects365 for the
detection checkpoints listed as `obj2coco`), which keep their own terms.

**Objects365 caveat.** The D-FINE README states that checkpoints trained or
pretrained on Objects365 (`*_obj365`, `*_obj2coco`) may be subject to the
Objects365 dataset terms and should not be assumed to be commercially cleared
under the D-FINE licence. The default detection weights of nitid (`nitid1s`,
`nitid1m`, `nitid1l`, `nitid1x`) are `obj2coco`. If you need a checkpoint
without this caveat, use the COCO-only variants. nitid has not established what
the Objects365 terms allow; review them before commercial use.

## Nitid Studio frontend (JavaScript bundle)

The Nitid Studio single-page app (`studio/`, built into the wheel under
`nitid/studio/static/`) bundles the following MIT-licensed libraries. The
copyright lines below were read from the `LICENSE` files of the installed
packages (versions locked in `studio/pnpm-lock.yaml` on 2026-11-18). Their MIT
permission notice is reproduced once, after the list.

- React 19.3.1 and React DOM 19.3.1 (`react`, `react-dom`),
  Copyright (c) Meta Platforms, Inc. and affiliates.
  Source: https://github.com/react/react
- Scheduler 0.28.0 (`scheduler`, a dependency of React DOM),
  Copyright (c) Meta Platforms, Inc. and affiliates.
  Source: https://github.com/react/react
- React Router 8.5.2 (`react-router`),
  Copyright (c) React Training LLC 2015-2019;
  Copyright (c) Remix Software Inc. 2020-2021;
  Copyright (c) Shopify Inc. 2022-2023.
  Source: https://github.com/remix-run/react-router

Source URLs are the `repository` fields of the installed packages' `package.json`.
[NV: the URLs were not fetched from here; only the package metadata and `LICENSE`
files were read.]

`react-router` also depends on `cookie-es` (MIT, Copyright (c) Pooya Parsa and
Roman Shtylman) and `@remix-run/route-pattern` (MIT, Copyright (c) 2025
Shopify Inc.). No identifiable code from either was found in the scaffold's
production bundle (checked on 2026-11-18 by searching the built JavaScript for
their names and characteristic identifiers; this is a heuristic, not a
guarantee). If a later build includes them, list them here. The remaining
production packages (`csstype`, `@types/react`, `@types/react-dom`) are
type-only and add no code to the bundle.

MIT License. Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions: the
above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS
IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The bundle's other development-time tooling (Vite, Rollup, TypeScript and
others) is not shipped in the wheel. `scripts/check-licenses.mjs` guards the
production dependency tree (see `docs/licensing/studio-weights.md`).

## ONNX Runtime Web

Added with the browser engine. This entry is a placeholder: the licence,
copyright holder and any additional notices of ONNX Runtime Web (and of the
`.wasm` binaries it loads at run time) must be read from the package that is
actually installed when the engine work package lands, and recorded here
before the first release that contains it. [NV: not verified; `microsoft/onnxruntime` was not
reachable from this audit session.]
