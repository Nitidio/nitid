# ADR-0004: Public model naming (`nitid<version><size>`)

- **Status**: Proposed
- **Date**: 2026-11-09
- **Deciders**: Jorge Abrines (Vaelsys)
- **Related**: [`THIRD_PARTY_NOTICES.md`](https://github.com/Nitidio/nitid/blob/main/THIRD_PARTY_NOTICES.md), README model tables

## Context

Up to 0.9.4 the public model names were `model1n`, `model1s`, `model1m`, `model1l` and `model1x`. The
prefix says nothing about the project, so every tutorial, issue and paper that quotes a model name
carries no brand. The names also did not follow the convention of the tools the main audience already
uses: engineers who come from Ultralytics pick a size (`n/s/m/l/x`) without having to know which
architecture is underneath.

Facts that constrain the choice:

- The models are D-FINE (detection) and D-FINE-seg (segmentation) checkpoints, both Apache-2.0.
  Attribution is required by the licence and is already carried by `THIRD_PARTY_NOTICES.md`, the root
  `LICENSE`, and the README model tables. A name does not remove that obligation.
- nitid does not redistribute pretrained weights; checkpoints are downloaded on first use from the
  sources their authors publish. If that ever changes, every weight repository must ship the upstream
  `LICENSE` and `NOTICE`, not only a sentence in a model card.
- The architecture may change in the future. The name must survive that without breaking users.
- Benchmarks are cited by hardware vendors and researchers, who need to map a nitid name to the
  upstream architecture and checkpoint.

## Options considered

1. **Keep `model1*`.** No migration cost. No brand, and the prefix is generic.
2. **Use the architecture name (`d-fine-m`) as the primary name.** Fully transparent, but it forces the
   user to learn D-FINE before choosing a size, and every mention reinforces the upstream project's
   brand rather than nitid's. Right only if the audience were mainly researchers.
3. **`nitid<version><size>` as the primary name, architecture name as a technical alias.** Chosen.
   The version number is what protects users if the architecture changes: a new architecture ships as
   `nitid2*` and `nitid1*` keeps meaning what it meant.

## Decision

- The canonical public name is `nitid<version><size>`, for example `nitid1s`. The hyphenated spelling
  `nitid1-s` and any case are accepted and normalised to the canonical name.
- `d-fine-<size>` (also `dfine_<size>`) is accepted as a technical alias and resolves to the current
  generation of that size (`nitid1<size>`). It cannot carry a version.
- The legacy `model1<size>` names keep working and resolve to `nitid1<size>`. They are documented as
  legacy and may be removed in a later minor release.
- `NITID.nitid_model` and every message print the canonical name, whatever spelling was given.
- Attribution stays visible: the README model tables list the upstream architecture next to each name,
  and `THIRD_PARTY_NOTICES.md` keeps the licence and copyright notices.

## Consequences

- One name per model in docs, tutorials and benchmark output; reference tables are keyed by canonical
  name, and the OpenVINO benchmark tool normalises legacy and alias spellings before the lookup.
- A `nitid1<size>` name must stay bound to one architecture and one set of default checkpoints. If a
  default checkpoint changes (for example COCO-only to Objects365+COCO, 48.5 → 50.7 mAP for S), the
  change is recorded in the changelog and the README table, because the name alone would otherwise hide
  it.
- `nitid1n` for detection is still unresolved: the code rejects it because no official DFINE-N
  detection checkpoint is registered, while the upstream README lists one. Revisit separately.
- Revisit if the architecture changes (new version number), if weights start being redistributed (add
  `LICENSE`/`NOTICE` per weight repository), or if the audience shifts towards researchers.
