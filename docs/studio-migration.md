# Migrating to Nitid Studio 0.10

Nitid Studio used to live in a separate repository, `Vaelsys/nitid-studio` (a Next.js website with a
browser demo), and was reached through a website and a desktop installer. From 0.10.0 it is one app
inside the `nitid` library. This page tells you what changes and what to do, depending on who you are.

!!! info "Why this changed"
    Two repositories meant two release trains and a REST contract documented on only one side.
    The reasoning, options and consequences are in [ADR-0005](adr/0005-nitid-studio.md).

## If you used the old site or the desktop installer

1. Install the library with the new extra and start the app:

    ```bash
    pip install "nitid[studio]"
    nitid studio
    ```

    It prints a local URL with a one-time token. The server listens on `127.0.0.1` only.
2. Pick **Local** (PyTorch or OpenVINO on your machine) or **Browser** (ONNX Runtime Web in the tab;
   nothing leaves your device).
3. Your exports keep working: PNG, JSON, CSV, YOLO TXT, crops and PDF are still there; a ZIP bundle
   was added.

| Old | New |
| --- | --- |
| Website demo (`/demo`) | `nitid studio`, route `/run/browser` |
| "Server jobs" panel (`/studio/server`) with a separate `nitid` backend and accounts | `nitid studio`, route `/run/local`; no accounts, no extra backend to start |
| LibreYOLO tasks (pose, depth, OBB, restoration, ...) | Not included; the Studio covers `nitid1*` detection and instance segmentation plus your own ONNX |
| Desktop installer | Superseded by `nitid studio`. [NV: the installer's fate is decision D4; the old one is retired or marked obsolete, not updated] |

## If you consumed the old REST backend

The `/api/v1` API is new and documented in [Nitid Studio](studio.md#api-v1) and in
[`studio/api-contract/`](https://github.com/Nitidio/nitid/tree/main/studio/api-contract). It does not
reproduce the old demo backend (no accounts; a bearer token printed at start-up instead). Update your
client to the new endpoints; breaking changes later will go to `/api/v2`.

## If you contributed to the old repository

- Work now happens in `Nitidio/nitid`: backend in `nitid/studio/`, frontend in `studio/`.
  See [Studio development](studio-development.md).
- The history was not imported; what was taken and what was dropped is recorded in
  [`studio/IMPORT.md`](https://github.com/Nitidio/nitid/blob/main/studio/IMPORT.md).
- New browser-runnable models are not added by extending a proxy allow-list any more: a model enters
  through `studio/models.lock.json` (revision, SHA-256, licence) and the rules in
  [Studio weights](licensing/studio-weights.md).

## Decisions taken at acceptance (2026-11-19)

Recorded by the project owner when [ADR-0005](adr/0005-nitid-studio.md) moved to *Accepted*.
Each one is reversible unless noted.

| # | Decision | Outcome |
| --- | --- | --- |
| D4 | Name and fate of the old site, the installer and the old repository | The product is called "Nitid app" for people and `nitid studio` / `nitid[studio]` in commands and packages. [NV: the owner answered "Nitid app"; read as the user-facing name of this single app, command and package names unchanged]. The old installer is retired or marked obsolete; the marketing site and the old repository are archived or moved once the import is merged |
| D5 | History | Curated snapshot with `studio/IMPORT.md`, no history |
| D7 | `model1*` aliases | Kept until 0.11 (they already warn with `DeprecationWarning`) |
| D8 | `nitid1n` detection | Registered with a "COCO only" notice (`dfine_n_coco.pth`, SHA-256 verified) |
| D9 | Public demo on Cloud Run | Untouched; revisit after the hosted phase |

## What is not in 0.10.0

- The **hosted** deployment (browser-only, served statically): blocked on a hosting decision and on the
  licence checks in [Studio weights](licensing/studio-weights.md).
- WebGPU in the browser engine: not validated on a real GPU.
- Video input.
