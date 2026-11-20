# Nitid Studio: hosted model bundle

Work package WP-6.2. The hosted Studio (`vite build --mode hosted`) is a static site with no
backend: the in-browser engine downloads ONNX files from a folder served **from the same origin as
the UI** (decision of the owner: Cloudflare, `app.nitid.io`; `url` values in the catalogue are
therefore relative file names). This page explains how that folder is produced, what is in it, how
the licence gate works and how big the files really are.

Marks: **[V]** measured or read in the session that wrote this page (2026-11-20); **[NV]** not
verified, with what is missing.

## In one paragraph

The pipeline reads [`studio/models.lock.json`](https://github.com/Nitidio/nitid/blob/main/studio/models.lock.json)
(the licence-audited list of weights), keeps the entries with `hostable: true`, downloads each
pinned checkpoint, **checks its SHA-256 against the lock before loading it**, exports it with the
same exporter the local Studio server uses (`nitid.studio.onnx_service.export_checkpoint`: static
`[1, 3, S, S]`, post-processed, simplified, `nitid_meta` embedded) and writes the folder below.
A second tool re-reads the folder and checks it. Nothing is uploaded anywhere by these tools.

## Layout

```text
models.json                       catalogue read by studio/src/browser/catalog.ts (version 1)
nitid1s-detect.onnx               one ONNX per (model, task); name = <id>-<task>.onnx
nitid1s-detect.onnx.meta.json     sidecar: sha256, size, source checkpoint, copy of nitid_meta
notices/Peterande-D-FINE/LICENSE  upstream licence files named by the lock's notice_files
NOTICE.txt                        attribution + "modified by exporting to ONNX" notice
SHA256SUMS                        sha256sum -c compatible, every file except itself
```

`models.json` has exactly the keys the browser parser accepts:

```json
{ "models": [ { "id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx",
                "sha256": "<64 hex>", "label": "nitid1s (detect, coco)" } ] }
```

`sha256` is the browser's cache key and integrity check, so the pipeline always writes it. The
sidecar is for humans and tools (it says which lock entry, URL and revision the file came from);
the browser does not read it. No `.pth`/`.pt` file is ever written to the bundle.

## The licence gate

- Only `hostable: true` entries are published. Asking for a `hostable: false` entry by lock id
  (every `obj2coco` entry, for example `--models dfine-s-obj2coco`) is an **error**, not a silent
  skip, and nothing is written. Asking for a public name (`--models nitid1s`) selects only its
  hostable entries and fails when it has none.
- The entry needs `sha256_status: "verified"`, no `open_caveats` and its `notice_files`.
- The licence texts come from [`studio/notices/`](https://github.com/Nitidio/nitid/tree/main/studio/notices)
  (`<owner>-<repo>/<file>`), where the upstream `LICENSE` files read during the audit are
  committed. A missing or empty file stops the run: no licence file, no publication.
- `hostable` is the result of a mechanical gate, not a legal opinion; the decision to host stays
  with the owner (see [Licence audit](licensing/studio-weights.md)).

## Producing the bundle

```bash
uv sync --extra dev                                   # onnx, onnxsim, jsonschema
uv run python -m tools.publish_models --out dist-models                 # n and s (default); NOTE: the hosted site can only serve n (25 MiB per-file limit), use --sizes n
uv run python -m tools.check_hosted_bundle dist-models                  # validate
```

| Option | Meaning |
| --- | --- |
| `--out DIR` | Bundle folder (required). |
| `--sizes n,s,m,l,x` | Size allow-list. Default `n,s` (small sizes until sizes are measured) unless `--models` is given. |
| `--models a,b` | Lock ids (`dfine-s-coco`) or public names (`nitid1s`). |
| `--tasks detect,segment` | Default: both. |
| `--offline-from DIR` | Use weights already on disk, named like the release assets (`dfine_s_coco.pth`). They are still checked against the lock. |
| `--cache-dir DIR` | Keep downloaded weights between runs. |
| `--notices-dir DIR` | Where licence texts are read from (default `studio/notices`). |
| `--force` | Rebuild even if up to date. |
| `--dry-run` | Print the selection and stop. |

Properties worth knowing:

- **Idempotent.** If `<file>.meta.json` says the ONNX was built from the same source SHA-256 with
  the same nitid version and the file still has the recorded hash, it is reused without
  downloading or exporting. A second run changes nothing, byte for byte.
- **Deterministic.** Order is by size (n, s, m, l, x), then task, then lock id; JSON is written
  with fixed formatting. A forced rebuild of the tiny test checkpoints gave identical ONNX bytes
  **[V]**; on the same machine the real models were not rebuilt twice, so byte-for-byte
  reproducibility across torch/onnx versions is **[NV]**. Treat the ONNX SHA-256 as belonging to
  the toolchain that built it (see the table).
- **Stale files are pruned.** ONNX files and sidecars of earlier runs that are not in the current
  selection are removed from `--out`, so `SHA256SUMS` and the catalogue always describe the folder.
- `nitid_meta.model` is set from the lock when the exporter could not infer the name, and the run
  fails if an ONNX names a different model or task than the lock.
- Segmentation weights come from Hugging Face; they are built the same way but have **not** been
  built in the sandbox of this work package (Hugging Face is blocked there) **[NV]**.

Then serve the folder next to the UI and point the hosted build at it, for example
`STUDIO_MODELS_BASE_URL=/models/ pnpm run build:hosted` with the bundle copied to `models/`
(see the [frontend guide](studio-development.md)).

## What the validator checks

`python -m tools.check_hosted_bundle DIR [--lock FILE] [--max-total-bytes N]` re-reads the folder and
trusts nothing the publisher wrote. It prints a size report and exits 1 on any problem:

- `models.json` has the shape `parseHostedCatalog` accepts, non-empty, with a `sha256` and a
  plain relative `.onnx` file name per entry (no scheme, no path);
- every `sha256` matches its file and every `.onnx` is in the catalogue;
- every ONNX has a `nitid_meta` valid against
  [`onnx-meta.schema.json`](https://github.com/Nitidio/nitid/blob/main/studio/api-contract/onnx-meta.schema.json),
  consistent with the catalogue (task, model, static square input) and with the sidecar copy;
- every sidecar names a lock entry that is `hostable: true` with a verified sha256 and the same
  source hash as the lock;
- the licence files each entry requires, and `NOTICE.txt`, are present;
- no `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.torchscript`, `.engine` or `.bin` file;
- `SHA256SUMS` lists exactly the files present and every hash matches.

The contract with the browser is tested on both sides: `tests/unit/test_publish_models.py` proves
the catalogue the tool writes is byte-identical to
`studio/src/browser/__fixtures__/hosted-models.json`, and `studio/src/browser/hosted-bundle.test.ts`
parses that fixture with the real `parseHostedCatalog`. To check a real bundle:

```bash
cd studio && NITID_HOSTED_CATALOG=../dist-models/models.json pnpm exec vitest run hosted-bundle
```

## Measured sizes (risk R13)

Detection, COCO-only checkpoints from the `dfinev1.0` release (the audited, pinned URLs), exported
at the default size (640 x 640), batch 1, static, post-processed, simplified. Built with nitid
0.9.6, torch 2.13.0, onnx 1.20.1, onnxruntime 1.28.0 on 2 CPUs and 7 GB of RAM, 2026-11-20.

| Model | Checkpoint `.pth` | ONNX bytes | ONNX MiB | ONNX SHA-256 | Build time / peak RSS |
| --- | ---: | ---: | ---: | --- | --- |
| `nitid1n` | 15,489,558 | 15,383,183 | 14.67 [V] | `ef561cd56d4ad0164a6ea326b3f37e4ef6d424f83b75271dfbea7ba79c0907ba` | about 12 s with download [V] / not recorded |
| `nitid1s` | 41,841,422 | 41,638,053 | 39.71 [V] | `a0dd8aee495f81c7ce6e745b00ff2071887a5109216dced6fc9377e0ea920c7f` | 17 s / 1.6 GB [V] |
| `nitid1m` | 79,108,938 | 77,708,250 | 74.11 [V] | `8c67d2d61cb8c657ce14d8ece3fe8bbf3ac6e6005e29739615779571d5ac66fd` | 26 s / 2.1 GB [V] |
| `nitid1l` | 126,059,754 | 123,804,142 | 118.07 [V] | `29a489d3488174c4b06328185b18f9a9616a051a19458f323bcf0d8074ddaaf0` | 43 s / 2.3 GB [V] |
| `nitid1x` | 252,128,986 | 247,641,727 | 236.17 [V] | `9bc94f3e02a4ed8fc91cc0e4f20dd8e31f6b77d3d7aced914201be5fac84d88b` | under 3 min (whole 5-model run took about 3.5 min) / not recorded |

The five ONNX files together are 506,175,355 bytes; the complete bundle with sidecars, notices and
checksums is 506,206,348 bytes (482.76 MiB) **[V]**. The validator accepted this bundle
(`hosted bundle OK`), `sha256sum -c SHA256SUMS` was clean, and the TypeScript parser accepted its
`models.json` **[V]**. Sizes in bytes are the ones the user's browser downloads if nothing
compresses them; ONNX weights compress poorly and compression was **not measured** **[NV]**.
The ONNX is slightly smaller than the checkpoint because the checkpoint stores EMA weights in
addition to the model **[NV: reason not checked]**.

Real inference check **[V]**: each of the five ONNX files loaded in ONNX Runtime 1.28.0 (CPU
provider), read its own `nitid_meta`, and on `docs/assets/demo/street.jpg` (squash-resized to
640 x 640, RGB, 0-1) returned `labels`, `boxes`, `scores` (the meta says `topk` 300); with score above 0.5 the
models returned 18 (n), 24 (s), 23 (m), 29 (l) and 23 (x) detections, mostly `person` and
`traffic light` (the image was not inspected by eye). This confirms the files load and detect; it is
**not** an accuracy evaluation, and not a check of the browser engine (ONNX Runtime Web / WASM).

Consequences to decide, not decided here:

- **Per-file limit [V, Cloudflare changelog of 2025-09-02, may have changed since]:** Workers static
  assets allow 25 MiB per file. Only `nitid1n` (14.67 MiB) fits; `nitid1s` (39.71 MiB) and larger
  do not. Serving s/m/l/x from the same origin needs R2 (or another store) fronted by the same
  hostname, or a different split of the files. The current default `n,s` therefore does not fit as
  static assets of a Worker. Check the current limit before choosing.
- **First-use download:** s is about 40 MiB and x about 236 MiB per visitor on first use (then
  cached by the browser by SHA-256). Whether that is acceptable on mobile is a product decision.
- **Not measured:** WASM inference time and memory in a browser for any size; segmentation
  exports (n, s, m, l, x) of any size.

## Tests

| File | What it covers |
| --- | --- |
| `tests/unit/test_publish_models.py` | Selection, size/task/model filters, refusal of non-hostable entries, lock gate, notices, catalogue shape and the TypeScript fixture, checksums, idempotence, pruning, offline and cache behaviour, CLI. |
| `tests/unit/test_check_hosted_bundle.py` | Every validator rule, passing and failing. |
| `tests/integration/test_publish_models.py` | Tiny random checkpoints (detect and segment) through the real exporter: bundle, validator, ONNX Runtime run, byte-identical rebuild, CLI, and the vitest parser check (skipped without `studio/node_modules`). |
| `studio/src/browser/hosted-bundle.test.ts` | `parseHostedCatalog` on the Python-written catalogue. |

Related: [Licence audit](licensing/studio-weights.md), [Studio frontend guide](studio-development.md),
[Export](export.md) (the `nitid_meta` format).
