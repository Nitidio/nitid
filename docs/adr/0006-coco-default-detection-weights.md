# ADR-0006: Default detection weights are COCO-only

- **Status**: Accepted (2026-11-19, by the project owner)
- **Date**: 2026-11-19
- **Deciders**: Jorge Abrines (Vaelsys)
- **Related**: [ADR-0004](0004-public-model-naming.md),
  [`THIRD_PARTY_NOTICES.md`](https://github.com/Nitidio/nitid/blob/main/THIRD_PARTY_NOTICES.md),
  [`docs/licensing/studio-weights.md`](../licensing/studio-weights.md)

Amends ADR-0004, which records `obj2coco` as the default detection weights.

## Context

- Until 0.9.6, `nitid1s/m/l/x` (detection) downloaded the Objects365+COCO checkpoints (`obj2coco`) by
  default. The D-FINE README says checkpoints trained or pretrained on Objects365 "may be subject to
  the Objects365 dataset terms and should not be assumed to be commercially cleared" under the
  D-FINE licence.
- nitid has not established what the Objects365 terms allow for derived models. [NV: the dataset terms
  were not read by the project.] The COCO-only checkpoints (`coco`) carry no such note.
- Published COCO val mAP (D-FINE README): COCO-only 48.5 / 52.3 / 54.0 / 55.8 versus Objects365+COCO
  50.7 / 55.1 / 57.3 / 59.3 for S / M / L / X, that is 2.2 to 3.5 points less. The nano model exists
  only as COCO-only (42.8).
- The default is what most users get without choosing, so it is the exposure that matters.

## Options considered

1. **Keep `obj2coco` as the default and document the caveat (option A).** Best accuracy; leaves every
   default user on weights of unknown commercial status. Rejected by the owner as imprudent.
2. **Make `coco` the default and keep `obj2coco` opt-in with a notice (option B).** Chosen.
3. **Remove `obj2coco` altogether.** Cleanest legally; loses accuracy for users who accept the risk and
   breaks anyone who asks for it explicitly. Rejected.

## Decision

`DEFAULT_WEIGHTS = "coco"`. `weights="obj2coco"` still works for `s/m/l/x`, downloads to
`dfine_<size>_obj2coco_wrapped.pth`, and prints a notice about the Objects365 caveat. `weights="default"`
means `coco`. `nitid1n` has no `obj2coco` variant, so asking for it is an error. The `obj2coco` entries
stay `hostable: false` in `studio/models.lock.json`.

Reversible: setting `DEFAULT_WEIGHTS` back to `"obj2coco"` restores the old behaviour for `s/m/l/x`
(the nano model would then need a fallback to `coco`).

## Consequences

- Default users lose 2.2 to 3.5 published mAP points on `nitid1s/m/l/x`; the README table now shows the
  COCO-only figures and lists the opt-in ones.
- Existing `*_obj2coco_wrapped.pth` files keep working when passed by path; `NITID("nitid1s")` now
  downloads `dfine_s_coco_wrapped.pth` instead of reusing an old default file.
- Fine-tuning from the default weights starts from COCO-only weights, which the D-FINE authors consider
  less general than the Objects365 ones for fine-tuning; users who want them can pass
  `weights="obj2coco"` and read the notice.
- Revisit if the Objects365 terms are read and found compatible.
