# Nitid Studio: licence audit of weights and dependencies

Work package WP-0.2 (issue #37). Audit date: **2026-11-18**. Companion files:
[`studio/models.lock.json`](https://github.com/Nitidio/nitid/blob/main/studio/models.lock.json) (machine-readable pins),
[`studio/models.lock.schema.json`](https://github.com/Nitidio/nitid/blob/main/studio/models.lock.schema.json),
[`tools/check_models_lock.py`](https://github.com/Nitidio/nitid/blob/main/tools/check_models_lock.py),
[`studio/scripts/check-licenses.mjs`](https://github.com/Nitidio/nitid/blob/main/studio/scripts/check-licenses.mjs) and
[`THIRD_PARTY_NOTICES.md`](https://github.com/Nitidio/nitid/blob/main/THIRD_PARTY_NOTICES.md).

This document records **facts and where they were read**. It is not legal advice; the decision
to host anything is Jorge's. Marks: **[V]** verified in this audit from the source named in the
row; **[NV]** not verified here, with what would have to be checked.

## How the facts were obtained

The environment blocks `huggingface.co` and GitHub web pages, and `gh api` refused repositories
that were not attached to the session. What worked, and what each fact below relied on:

- Anonymous `git clone --depth 1` of the upstream repositories (read access through the session
  proxy): `LICENSE` files, README licence statements, commit ids.
- `git ls-remote` on `Peterande/storage` for the release tag.
- `curl` against the release asset URLs: sizes (HTTP `Content-Range`), and the full download of
  the three assets under 50 MB to compute SHA-256 in `/tmp` (not committed). The larger assets
  were hashed later with `tools/hash_models_lock.py` (see below).
- Package metadata and `LICENSE` files of the installed frontend dependencies (`pnpm install
  --frozen-lockfile` in `studio/`).

Not reachable: `huggingface.co`, the GitHub REST releases API for `Peterande/storage`
(so release ids and dates are unknown), and `microsoft/onnxruntime`.

## Finding: D-FINE licence status (the internal note and ADR-0004 disagree)

The internal note said the D-FINE licence was "not verified"; ADR-0004 says D-FINE and
D-FINE-seg are Apache-2.0. What was checked:

1. **[V] Code licence.** `LICENSE` in `Peterande/D-FINE` (commit
   `956d1709314c2c6a4df6f34de232054578a7449f`), in `Peterande/storage` (the repository that hosts
   the checkpoints, `master` at `7f91133e7a5d5713c64e6de0892b3599b4f5646f`) and in
   `ArgoHA/D-FINE-seg` (commit `13385abc7ea1e0ec290d1e06b4565942bd76efa4`) are byte-identical
   (11357 bytes, SHA-256 prefix `c71d239df91726fc`) and are the Apache License 2.0 text. The
   D-FINE README carries an "Apache 2.0" licence badge.
2. **[V] The Apache-2.0 statement is not the whole story for the Objects365 checkpoints.** The
   D-FINE README (same commit, "Notes" under the model tables) says that checkpoints trained or
   pretrained on Objects365 (`*_obj365`, `*_obj2coco`) *may be subject to the Objects365 dataset
   terms and should not be assumed to be commercially cleared under the D-FINE license*. ADR-0004
   and `THIRD_PARTY_NOTICES.md` state "Apache-2.0" for all D-FINE checkpoints without this
   caveat. **This mattered because `DEFAULT_WEIGHTS` was `"obj2coco"`** in `dfine/utils/downloads.py`
   up to 0.9.6, so `nitid1s/m/l/x` for detection resolved to `obj2coco` checkpoints by default
   (changed in 0.10.0, see below). For the coco-only variants (`weights="coco"`) the README makes
   no such statement.
3. **[NV]** Whether the Objects365 terms actually restrict commercial use of the derived weights
   was not researched here (the dataset terms were not read). The upstream authors' own wording is
   cautious, and so is this repository: obj2coco entries are `hostable: false`.
4. **[NV]** No `NOTICE` file exists in any of the four repositories read (D-FINE, D-FINE-seg,
   RT-DETR, DEIM; root listing checked). DEIM's `LICENSE` carries an extra copyright line for
   INTELLINDUST (already recorded in `THIRD_PARTY_NOTICES.md`).

Follow-up **option A (2026-11-19, morning)**: `THIRD_PARTY_NOTICES.md`, ADR-0004 and the README
stated the caveat and kept `obj2coco` as the default. **Superseded the same evening by option B
(owner's decision, "seamos prudentes", [ADR-0006](../adr/0006-coco-default-detection-weights.md)):**
`DEFAULT_WEIGHTS` is now `"coco"`; `obj2coco` is opt-in, prints the caveat when downloaded, and its
entries stay `hostable: false`. Reopen the question if the Objects365 terms are read and found
compatible (clear `open_caveats`, optionally restore `obj2coco` as the default with a new ADR).

## Table: weights and upstream

| Model / weights | Upstream | Licence (SPDX) | Exact source | Verified | Can Studio host a derived ONNX? |
| --- | --- | --- | --- | --- | --- |
| `nitid1n` detect, `dfine_n_coco.pth` (not registered in nitid today) | `Peterande/D-FINE` | Apache-2.0 | `LICENSE` at D-FINE `956d170`; asset `https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth` (15 489 558 B) | **[V]** licence, size, SHA-256 | Passes the gate (`hostable: true`). Needs the checklist below. |
| `nitid1s` detect, `dfine_s_coco.pth` | idem | Apache-2.0 | asset `.../dfine_s_coco.pth` (41 841 422 B) | **[V]** licence, size, SHA-256 | Passes the gate (`hostable: true`). |
| `nitid1m/l/x` detect, `coco` variant | idem | Apache-2.0 | assets `dfine_{m,l,x}_coco.pth` (79 108 938 / 126 059 754 / 252 128 986 B) | **[V]** licence, size, SHA-256 (computed 2026-11-19 with `tools/hash_models_lock.py`) | Passes the gate (`hostable: true`). Needs the checklist below. |
| `nitid1s` detect, `dfine_s_obj2coco.pth` (opt-in) | idem | Apache-2.0 for the code; **Objects365 caveat open** | asset `.../dfine_s_obj2coco.pth` (41 847 790 B) | **[V]** licence, size, SHA-256, caveat text; **[NV]** what the Objects365 terms allow | No (`open_caveats`). |
| `nitid1m/l/x` detect, `obj2coco` (opt-in) | idem | same | assets `dfine_m_obj2coco.pth`, `dfine_l_obj2coco_e25.pth`, `dfine_x_obj2coco.pth` (79 117 378 / 126 083 766 / 252 140 594 B) | **[V]** licence, size, SHA-256 (2026-11-19), caveat; **[NV]** Objects365 terms | No (`open_caveats`). |
| `nitid1*` segment, `dfine_seg_{n,s,m,l,x}_coco.pt` | `ArgoHA/D-FINE-seg`, weights on `huggingface.co/ArgoSA/D-FINE-seg` at commit `79572c466448db97aa2b2f029c907d8e6b2a42fc` | Apache-2.0 for the COCO checkpoints (model card paragraph and upstream README); Hub licence tag is `other` | Model card at the pinned commit; README at D-FINE-seg `13385ab`; code `LICENSE` same commit | **[V]** card paragraph, README, code licence, commit, sizes, LFS SHA-256 (reported by the Hub API, 2026-11-19); **[NV]** SHA-256 recomputed from a download | Passes the gate (`hostable: true`), with the conditions below (no LICENSE/NOTICE in the Hub repository; tag `other`). Lock ids `dfine-seg-{n,s,m,l,x}-coco`. |
| D-FINE-seg Cityscapes semantic weights (`dfine_sem_*_cityscapes.pt`) | `ArgoHA/D-FINE-seg` | Non-commercial (derivative of Cityscapes, per the README) | README `License` section, same commit | **[V]** README statement | **No. Excluded.** |
| D-FINE-seg keypoints (`dfine_kpts_s_coco.pt`) | idem | Apache-2.0 per the README (COCO weights) | same | **[V]** README; **[NV]** Hub | Out of scope (pose is not in Studio). |
| RT-DETR (code lineage only) | `lyuwenyu/RT-DETR` | Apache-2.0 | `LICENSE` at `df2a96c8c3a098567a1149bf0248da6efe6d495d` | **[V]** | No weights involved. |
| DEIM (training recipe attribution only) | `ShihuaHuang95/DEIM` | Apache-2.0 | `LICENSE` at `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` (has an INTELLINDUST copyright header) | **[V]** | No weights involved. |
| DETR, PaddleDetection HGNetv2 (code lineage) | `facebookresearch/detr`, `PaddlePaddle/PaddleDetection` | Apache-2.0 per `THIRD_PARTY_NOTICES.md` | not re-read here | **[NV]** (existing entries, not re-audited) | No weights involved. |
| ONNX Runtime Web | `microsoft/onnxruntime` | not read | not read | **[NV]**: read the licence and third-party notices of the installed npm package when the browser engine lands | Not weights; a runtime dependency (see the dependency section). |

Revision pinned for the D-FINE checkpoints: GitHub release tag `dfinev1.0` of
`Peterande/storage` (a tag that points at commit `39a3115d4b1183b9035e333e57655f018272ebbe`; a
second tag `dfinev1` also exists). GitHub release assets can be replaced under the same tag, so
the **SHA-256 is the real integrity pin**, not the tag.

### Excluded by policy (never bundled or hosted)

These come from the project rules for Studio (ADR-0005 decision 5 and the work-package brief). Their
licences were **not** re-read in this audit, so each is **[NV]**; they are listed so nobody adds them
by accident, and `tools/check_models_lock.py` would reject every one of them because none is in the
licence allowlist.

| Item | Reason given | Status |
| --- | --- | --- |
| The six "Libre*" ONNX files in `Vaelsys/nitid-studio` | Provenance not documented (ADR-0005) | **[NV]**: would need the upstream repo, licence and checkpoint of each file |
| FOMO, LocateAnything, DexiNed/TEED, LeVJEPA | Provenance or licence not documented | **[NV]** |
| RF-DETR XL / 2XL | Brief lists them under a platform model licence (PML) rather than an open one | **[NV]**: read the licence in the weights' source |
| Ultralytics YOLO (AGPL-3.0), YOLOv9 (GPL-3.0), YOLO-NAS (non-commercial weights) | Copyleft or non-commercial | **[NV]**: licences recalled from the brief, not re-read |
| LibreYOLO | Code is MIT; the weights have a licence **per weight repository** (some CC-BY-NC, per ADR-0005 `[NV]`) | **[NV]**: read the licence of each weight repository before any use |
| Pose and OBB models | Flows retired from the project | Out of scope |

## What must be true before Studio may host any ONNX

Hosting means serving an ONNX file (or a download link to a file you control) to third parties.
Tick every item; any miss means "do not host".

- [ ] The decision to host is recorded by the owner (Jorge) in an ADR or in this document. The
      gate below is mechanical and is not a legal opinion.
- [ ] The source checkpoint has an entry in `studio/models.lock.json` with `hostable: true`, and
      `python -m tools.check_models_lock` passes (licence in the allowlist, revision pinned,
      SHA-256 verified, notice files listed, no `open_caveats`).
- [ ] The licence was read **from the upstream source** (file, revision) and the statement
      covers the *weights*, not only the code. Dataset terms behind the weights were considered
      (the Objects365 caveat is the live example).
- [ ] The upstream `LICENSE` and, if one exists, `NOTICE` travel with the files, in the repository
      or bucket that serves the ONNX (ADR-0004: "every weight repository must ship the upstream
      `LICENSE` and `NOTICE`"). For Apache-2.0 the licence text itself also asks that recipients
      get a copy of the licence and that modified files carry a prominent change notice; the ONNX
      export is a modification, so a short `NOTICE`/README saying "derived from X by exporting to
      ONNX with nitid vN" is the usual way to do it.
- [ ] `THIRD_PARTY_NOTICES.md` and the model table in the README name the upstream architecture
      and checkpoint (attribution), and the wheel still ships `THIRD_PARTY_NOTICES.md`.
- [ ] The **ONNX file has its own SHA-256**, recorded as a new lock entry with `format: "onnx"`,
      the nitid version and export command in `notes`. The checkpoint hash does not describe it.
- [ ] The ONNX follows ONNX contract v1 and the `nitid_meta` descriptor (ADR-0005), including the
      class names and the source checkpoint id.
- [ ] No weights are committed to git (`studio/scripts/check-size.mjs` already forbids `.onnx`,
      `.pth`, `.wasm` in `studio/`). They are served from a release asset, bucket or the Hub.
- [x] The default-weights question is settled (ADR-0006): the default and the hosted set use the
      coco-only variants. `obj2coco` stays out of any hosted set until the Objects365 caveat is
      resolved in writing and `open_caveats` cleared.
- [ ] Any weight with a non-commercial, copyleft, platform or unknown licence stays out (see the
      exclusion table).

## What to check on Hugging Face (`ArgoSA/D-FINE-seg`)

The Hub is not reachable from the audit environment, so this part was done by a person.

**Model card licence paragraph [V, 2026-11-19, pasted by the maintainer from the Hub card]:** the
code and the COCO-pretrained checkpoints (detection, instance segmentation, person keypoints) are
Apache-2.0. The Cityscapes semantic-segmentation checkpoints (`dfine_sem_*_cityscapes.pt`) are
derivative works of the Cityscapes dataset and inherit its terms: non-commercial use only (academic
research, teaching, scientific publications, personal experimentation). A model trained on your own
data is yours to ship; the code stays Apache-2.0. This matches the upstream README (item above), so
the card repeats the separation between COCO weights (hostable) and Cityscapes weights (excluded).

**Result of the Hub query (2026-11-19, `tools/hash_models_lock.py --huggingface-only`, run by the maintainer):**

- Commit: `79572c466448db97aa2b2f029c907d8e6b2a42fc` (pinned in the lock and, since 2026-11-19, in
  nitid's own downloader; see "Findings" below).
- Hub licence tag: **`other`**, not `apache-2.0`. The card's text paragraph (above) is what says
  COCO checkpoints are Apache-2.0; the tag alone does not.
- **No `LICENSE` and no `NOTICE` file** in the repository at that commit (files: `.gitattributes`,
  `README.md`, `config.json`, `assets/*`, and the `.pt` weights). The licence statement lives only in
  the card. ADR-0004 requires the upstream `LICENSE` to travel with any weight repository we host,
  so it must come from the code repository `ArgoHA/D-FINE-seg` (audited at `13385ab`).
- Segmentation weights, size in bytes and LFS SHA-256 (the Hub reports the SHA-256 of the file
  content for LFS files; it was **not** recomputed from a download here):

| File | Size | SHA-256 |
|---|---|---|
| `dfine_seg_n_coco.pt` | 17 113 439 | `064477a85ea102cb2652797987aacecfc7e870a798afb4d848fa7b75b665fdb2` |
| `dfine_seg_s_coco.pt` | 48 119 167 | `07500e13824b82c9ee7422fd98736259e3bafcad65bf63f48d0c386a4ba987c1` |
| `dfine_seg_m_coco.pt` | 85 380 411 | `babb166b9cdcfaba475485bd0152aeeae8dc351561b9ec6f7b37542368800dde` |
| `dfine_seg_l_coco.pt` | 132 322 067 | `8806097678cb7184ce48992c28e314b8514e79ea2933d1cfeb1df967ff3dc5d7` |
| `dfine_seg_x_coco.pt` | 258 783 791 | `06d848ffb0ab58eb18e3500c55cfc0ffc3a2bc82baa172b944858f40afceed10` |

**Findings from the file list:**

1. The same repository also holds **detection** weights as `.pt` (`dfine_{n,s,m,l,x}_coco.pt` and
   `dfine_{s,m,l,x}_obj2coco.pt`) and `dfine_kpts_s_coco.pt` (person keypoints). They are repackaged:
   sizes differ by a few hundred bytes from the `Peterande/storage` `.pth` files, so their hashes are
   **not** interchangeable with the lock's detection entries. nitid downloads detection weights from
   `Peterande/storage`, so these are not in the lock. The `obj2coco` ones carry the same Objects365
   caveat; the card paragraph above does not address them.
2. The Cityscapes semantic weights (`dfine_sem_{n,s,m,l,x}_cityscapes.pt`) are present at this
   commit and stay excluded (non-commercial).
3. nitid's downloader used `https://huggingface.co/ArgoSA/D-FINE-seg/resolve/main/...`, a moving
   ref: a new upstream upload under the same file name would have changed what users get while the
   lock still said otherwise. **Resolved 2026-11-19 (owner's decision):** `dfine/utils/downloads.py`
   now downloads from `.../resolve/79572c46.../`, and a unit test checks that every segmentation URL
   equals the one in the lock. The downloader does not yet verify the SHA-256 of these files (the
   hashes come from the Hub API and have not been recomputed from a download), so a wrong value
   would block every user; add `sha256=` to the segmentation assets after recomputing them.

To recompute the hashes from a real download (recommended before hosting), on a machine that can
reach the Hub: `python tools/hash_models_lock.py --only dfine-seg-s-coco,dfine-seg-m-coco,...`.

## Frontend dependencies (npm)

- `studio/scripts/check-licenses.mjs` runs `pnpm licenses list --prod --json` and fails when a
  production package has a licence outside MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC,
  0BSD or BlueOak-1.0.0. Result on the scaffold (2026-11-18): 9 production packages, all MIT.
- `A OR B` expressions pass when one alternative is allowlisted, with a warning so the choice is
  recorded (for example a package under "MIT OR GPL" would pass as MIT). `A AND B` needs every
  part allowed. Missing or unknown licences fail.
- Covers: the production tree reported by pnpm, including transitive and type packages.
  Does **not** cover: devDependencies (not shipped), what is really in the bundle, the accuracy
  of a package's declared licence, `.wasm` and model files fetched at run time, or attribution
  text (kept by hand in `THIRD_PARTY_NOTICES.md`).
- Command for CI (from `studio/`, after `pnpm install --frozen-lockfile`):
  `node scripts/check-licenses.mjs`. Its tests: `node --test scripts/check-licenses.test.mjs`
  (not picked up by Vitest, whose `include` is `src/**`). `package.json` was not edited by this
  work package; a `check:licenses` script can be added there later.
- Models lock check for CI (repository root): `python -m tools.check_models_lock`. It needs the
  `jsonschema` package, which is neither a dependency of the nitid wheel nor in the `dev` extra
  of `pyproject.toml` (checked 2026-11-18; `uv.lock` does not mention it). Install it in the CI
  job or add it to the `dev` extra: without it the unit tests are skipped
  (`pytest.importorskip`) and the CLI exits with an explicit error.
- `THIRD_PARTY_NOTICES.md` now has entries for React, React DOM, Scheduler and React Router
  (copyright lines read from the installed `LICENSE` files) and a placeholder for ONNX Runtime
  Web, to be filled when the browser engine package lands. `cookie-es` and
  `@remix-run/route-pattern` (dependencies of React Router) were not detected in the scaffold's
  bundle; recheck with every build that changes routing.
- Development tooling includes MPL-2.0 (`lightningcss`) and CC-BY-4.0 (`caniuse-lite`) packages
  (per `docs/studio-development.md`); they are not shipped and are outside the production check.

## Reproducing the checks

```bash
# weights lock
python -m tools.check_models_lock
python -m pytest tests/unit/test_check_models_lock.py -q

# frontend licences (from studio/)
node scripts/check-licenses.mjs
node --test scripts/check-licenses.test.mjs

# SHA-256 of an upstream asset (example)
curl -sSL -o /tmp/dfine_s_coco.pth \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth
sha256sum /tmp/dfine_s_coco.pth   # 48a6c8cc...3e89
```

## Computing the SHA-256 (`tools/hash_models_lock.py`)

The lock's SHA-256 is the integrity pin of every weight. `tools/hash_models_lock.py` reads
`studio/models.lock.json`, downloads each `url`, streams it into a SHA-256 and writes a JSON report.
It uses only the Python standard library.

```bash
python tools/hash_models_lock.py --missing-only            # hash only what is not verified yet
python tools/hash_models_lock.py --only dfine-m-coco       # a single entry
python tools/hash_models_lock.py --huggingface-only        # ArgoSA/D-FINE-seg commit, sizes, LFS hashes (no downloads)
python tools/hash_models_lock.py --missing-only --update-lock   # also write hashes into the lock
```

Safety rules built into the script: a hash or size that disagrees with the lock is reported as a
`mismatch` (exit code 1) and never overwritten; `--update-lock` changes only `sha256` and
`sha256_status`, never `hostable`, `open_caveats` or the licence fields (a licence decision belongs
to a person); the report is rewritten after every file, so an interruption loses nothing. Tests:
`tests/unit/test_hash_models_lock.py` (real local HTTP server, truncated downloads, redirects,
mismatches).

**Result of 2026-11-19** (downloaded from the release URLs; every size matched `size_bytes`):

| Entry | SHA-256 |
|---|---|
| `dfine-m-coco` | `b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195` |
| `dfine-l-coco` | `f1d76b20c06610e03fa833ac9524337635e5e74204f22f4f8721174f19303d78` |
| `dfine-x-coco` | `9a82252f2d0426b2c24e99905c36c33a4ea850f247c247bda76fb5f76daea123` |
| `dfine-m-obj2coco` | `183caeb10b3266b62b9bc36917f8b7c0b10c14d3ed2dc4c9ce776ea14e3ec1d8` |
| `dfine-l-obj2coco-e25` | `c913aef6dc89c327c1a8b9e4ee5cf93eeeb3a1692ffe8ab40f1aa576ee6c3611` |
| `dfine-x-obj2coco` | `39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b` |

As a control, the script reproduced the already verified `dfine-s-coco` hash
(`48a6c8cc…3e89`) from a fresh download. With these hashes `dfine-{m,l,x}-coco` pass the gate
(`hostable: true`); the three `obj2coco` entries stay blocked by the Objects365 caveat only.
