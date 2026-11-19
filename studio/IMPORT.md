# IMPORT: what Nitid Studio took from `Vaelsys/nitid-studio`

This file is the record promised by [ADR-0005](../docs/adr/0005-nitid-studio.md) (decision 6): a
curated snapshot instead of the git history of the old repository. It says what was taken, what was
left behind and why, so nobody has to reverse-engineer it later.

- **Source reviewed**: `Vaelsys/nitid-studio`, branch `main`, commit `908d13fd476576a9c6df909b7e77679eeb0941d6`
  (merge of PR #46, 2026-09-11). Read-only; the old repository was not modified.
- **Method**: the old site was read for behaviour and contracts, and the Studio was written again on
  the stack chosen in the ADR (Vite + React + TypeScript, FastAPI). **No source file was copied
  verbatim** [checked 2026-11-19: none of the old file or component names, such as `createPdfReport`,
  `LiveDetectionDemo` or `model-proxy`, appear in `studio/src`, `nitid/` or `docs/`].
- **Git history**: not imported (decision D5, accepted 2026-11-19). The old repository stays available
  for reference until it is archived (decision D4).

## What carried over (as behaviour, re-implemented)

| Old feature (`nitid-studio`) | Where it lives now |
| --- | --- |
| D-FINE demo: upload, run in the browser with ONNX Runtime Web, draw boxes and masks | `studio/src/browser/`, `studio/src/engine/`, route `/run/browser` |
| Local-only "server jobs" panel against a `nitid` REST backend | `studio/src/local/`, route `/run/local`, backed by `nitid/studio/` (the backend now ships in the library) |
| Exports: image, JSON, CSV, YOLO TXT, crops, PDF report | `studio/src/exports/` (adds a ZIP bundle and polygon labels for segmentation) |
| Model list with task and size | `GET /api/v1/models`, `studio/models.lock.json` |
| Allow-listed model proxy | Not needed: weights come from the author's host on first use, or from the user's own ONNX |

## What was left behind, and why

| Left behind | Reason |
| --- | --- |
| LibreYOLO workbench and model families (pose, depth, restoration, OBB, ...) | Out of scope (ADR-0005 decision 5; `docs/roadmap.md`). Weights carry per-repository licences, some non-commercial |
| Pose and OBB flows | Retired from the project |
| ONNX files in `public/models/` with undocumented provenance | Provenance and licence unknown; not imported |
| Marketing pages (applications, articles, benchmarks, commercial, products, software) | Not part of the app; the public site is a separate concern (decision D4) |
| Next.js / vinext / Cloudflare Worker, Transformers.js, `jszip`, `jspdf` | Replaced by the Vite SPA and a small dependency set (`fflate`, a hand-written PDF writer) |
| Unrestricted `nitid-server` proxy | Replaced by a real backend with a per-launch token and Host/Origin checks |
| User accounts of the demo backend | The local Studio has no accounts and no telemetry |

## Reproducing this review

```bash
git clone https://github.com/Vaelsys/nitid-studio && cd nitid-studio
git checkout 908d13fd476576a9c6df909b7e77679eeb0941d6
```

Compare `app/demo/`, `app/studio/server/` and `app/api/` with `studio/src/` and `nitid/studio/`.
