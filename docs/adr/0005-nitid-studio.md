# ADR-0005: Nitid Studio, a web app inside the nitid library

- **Status**: Proposed
- **Date**: 2026-11-18
- **Deciders**: Jorge Abrines (Vaelsys)
- **Related**: [ADR-0004](0004-public-model-naming.md), [`docs/roadmap.md`](../roadmap.md),
  [`THIRD_PARTY_NOTICES.md`](https://github.com/Nitidio/nitid/blob/main/THIRD_PARTY_NOTICES.md),
  the API contract in [`studio/api-contract/`](../../studio/api-contract/README.md)

Reversibility legend used in the Decision section: **[R]** reversible at low cost (a later PR can change
it without breaking users); **[S]** sticky (public API, package layout or licence posture; changing it
needs a new ADR and a migration).

## Context

A separate repository, `Vaelsys/nitid-studio`, holds a Next.js/vinext website and browser workbench for
the nitid ecosystem. It runs D-FINE ONNX models client-side with ONNX Runtime Web and has a local-only
demo that talks to a `nitid` REST backend. Keeping it apart from the library has costs: two release
trains, a REST contract that is documented only on the frontend side, and a demo that needs accounts and
has no rate limiting because it was never meant to be exposed.

Facts that constrain the choice:

- `nitid` is at 0.9.6. The base wheel is small (about 0.18 MB, measured by the author [NV: not
  re-measured here]); heavy dependencies are optional extras (`train`, `track`, `openvino`, `wandb`, ...).
- ADR-0004 fixes the public names (`nitid1<size>`; `model1*` are legacy aliases that resolve to them) and
  states that **nitid does not redistribute pretrained weights**: checkpoints are downloaded on first use
  from the sources their authors publish. It also says that if that ever changes, **every weight
  repository must ship the upstream `LICENSE` and `NOTICE`**, not only a sentence in a model card.
- The detection and segmentation models are D-FINE and D-FINE-seg checkpoints, Apache-2.0, with attribution
  carried by `THIRD_PARTY_NOTICES.md`.
- `docs/roadmap.md` lists "becoming a general YOLO model zoo" as not planned unless an optional adapter is
  justified by measured demand.
- LibreYOLO (the model collection the existing site also exposes) has MIT code but weights under
  per-repository licences, some of them CC-BY-NC. [NV: stated by the project owner; the licence of each
  weight repository must be read before any use.]
- The existing site has pose and OBB flows that were retired from the project, and ONNX files whose
  provenance is not documented.
- A browser cannot run PyTorch; it can run ONNX Runtime Web, and multi-threaded WASM needs the page to be
  cross-origin isolated (COOP/COEP headers).

## Options considered

1. **Keep the separate repository and only document the REST contract.** No change to the library. Costs:
   the drift between library and UI that motivated this ADR stays; `pip install nitid` never gives the user
   a UI.
2. **Move the Next.js/vinext site in as-is, and ship its static export.** Reuses the code. Costs: the site
   is built for Cloudflare Workers with server routes (a model proxy and an unrestricted local proxy); a
   static export would drop exactly the parts that need a server, and it brings a Next/vinext toolchain and
   a LibreYOLO/pose/OBB model surface that conflicts with the roadmap. Rejected.
3. **Python-rendered UI (Jinja templates, htmx, Gradio, Streamlit).** One stack. Costs: no client-side
   (browser-only) mode, which is what makes a hosted deployment possible, and a weaker image-annotation
   experience. Rejected.
4. **A separate frontend repository or a separate PyPI package (`nitid-studio`).** Clean separation of
   toolchains. Costs: the contract drift returns, and two things must be released in lockstep. Rejected.
5. **One monorepo app: a Vite + React + TypeScript SPA with a FastAPI backend inside the `nitid` package,
   built once and deployable in two ways.** Chosen. The rejected-most-confidently option is 2: it looks
   like the cheapest path and carries the licence and scope problems of the old site into the library.

## Decision

1. **One app, two deployments. [S for "same frontend"; R for the hosted phase]**
   - *Local*: `nitid studio` starts a FastAPI server bound to `127.0.0.1` and serves the SPA. The SPA can
     use the server (PyTorch, OpenVINO) or run in the browser (ONNX Runtime Web).
   - *Hosted*: the same SPA served statically, browser-only. At start-up the SPA probes
     `/api/v1/health`; if it is absent it switches to browser mode.
   - The hosted deployment is a **later phase, after 0.10.0**. It is blocked on verifying the licence and
     notice obligations of every set of weights **before any ONNX is published** (see ADR-0004: nitid does
     not redistribute weights, and if it starts to, each weight repository needs the upstream `LICENSE` and
     `NOTICE`).
2. **Monorepo layout. [S]** `nitid/studio/` holds the backend (Python, inside the wheel); `studio/` holds
   the frontend sources (outside the wheel); the static build is copied to `nitid/studio/static/` and
   packaged. Never versioned: `node_modules`, binaries, `*.onnx`, `*.pth`, `*.wasm`.
3. **Frontend: Vite + React + TypeScript SPA. [R]** Discarded: static export of vinext/Next, Python
   templates, a separate repository or package.
4. **Backend: local FastAPI. [S for "no accounts, no telemetry"; R for the rest]** No accounts; a
   per-launch bearer token plus Host/Origin validation; a single worker; no telemetry. Dependencies live in
   a new optional extra `studio = [fastapi, uvicorn, python-multipart]`; the base package does not change.
5. **Models in scope. [R]** Only `nitid1*` (detect and segment) and the user's own ONNX with a JSON
   descriptor. No LibreYOLO code or weights, and no third-party weights are bundled, consistent with the
   roadmap and with the licence situation above.
6. **Migration from `Vaelsys/nitid-studio`. [R]** A curated snapshot with a `studio/IMPORT.md` that records
   what was taken and why; not the git history. Pose and OBB are dropped, and ONNX files without provenance
   are not imported.
7. **Target version 0.10.0. [S]** A new minor: new extra, new command (`nitid studio`) and a new public
   package (`nitid.studio`). The `model1*` aliases stay until 0.11 (ADR-0004 allows removal in a later
   minor).
8. **ONNX contract v1. [S once any ONNX is published; R before]**
   - Input `images`: float32, static `[1, 3, S, S]`, RGB, range 0-1, resize "squash"
     (`Resize((S, S))`, aspect ratio not preserved).
   - Outputs: `labels` int64 `[1, K]`; `boxes` float32 `[1, K, 4]`, xyxy in pixels of the model input;
     `scores` float32 `[1, K]`; and, for segment only, `masks` float32 `[1, K, S/4, S/4]`.
   - Top-K is 300; the default NMS IoU is 0.85.
   - The ONNX file carries no class names. They travel in a JSON descriptor (`meta`), whose canonical
     form is [`studio/api-contract/onnx-meta.schema.json`](../../studio/api-contract/onnx-meta.schema.json).
     `sha256` in the descriptor is added only by the REST API, never embedded in the ONNX.
   - The REST API is specified in [`studio/api-contract/openapi.json`](../../studio/api-contract/openapi.json)
     and checked by `tests/unit/studio/test_contract.py`. FastAPI will later generate the definitive
     document and CI will fail on divergence.

## Consequences

- **Wheel size** grows from about 0.18 MB to an estimated 4-10 MB once the static build is packaged
  [NV: estimate, to be measured when the first build exists]. The Python dependencies stay optional.
- **Two stacks** (Python and Node) in one repository: CI needs a Node job, the release process needs a
  frontend build step before `python -m build`, and contributors to the library are not forced to install
  Node (the build output is what gets packaged).
- **Cross-origin isolation**: multi-threaded WASM requires COOP/COEP headers. The local server can set
  them; any static host for the hosted deployment must as well, or browser mode falls back to a single
  thread.
- **Licence risk when hosting**: publishing ONNX exports or serving weights makes nitid a redistributor.
  Before that, each weight source needs its licence read and its `LICENSE`/`NOTICE` carried, as ADR-0004
  requires. Local mode keeps the current posture: weights are fetched from their authors on first use.
- **Public API surface grows** (`nitid studio`, `nitid.studio`, extra `studio`, REST v1, ONNX meta v1),
  which is why this is an ADR. Breaking changes to the REST API need `/api/v2`; to the meta,
  `schema_version` 2.
- **Security surface**: a local HTTP server exists now. The token and Host/Origin checks are the barrier
  against other web pages reaching it; the server must never bind to a non-loopback address by default.
- **Revisit** if the hosted deployment is dropped (then the browser mode can be removed), if a second model
  family enters scope (the descriptor and the model list already allow it), or if the wheel size becomes a
  problem (move the static build into a separate wheel pulled by the `studio` extra).
