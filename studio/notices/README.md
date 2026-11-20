# Upstream licence texts for hosted models

`tools/publish_models.py` copies these files into the hosted model bundle (`notices/<owner>-<repo>/`),
as required by the `notice_files` of each entry in `studio/models.lock.json`.

- `Peterande-D-FINE/LICENSE`: Apache-2.0, read from `Peterande/storage` (the host of the D-FINE
  checkpoints), `master` 7f91133e7a5d5713c64e6de0892b3599b4f5646f; byte-identical to the one in
  `Peterande/D-FINE` (see `docs/licensing/studio-weights.md`).
- `ArgoHA-D-FINE-seg/LICENSE`: Apache-2.0, `ArgoHA/D-FINE-seg` at commit 3e69af991ec60eb8adebc2b349e92c53694279e0 (cloned 2026-11-20);
  the audit recorded 13385abc7ea1e0ec290d1e06b4565942bd76efa4, and the text is byte-identical
  (11357 bytes, SHA-256 prefix c71d239df91726fc).
