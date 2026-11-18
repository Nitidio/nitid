# Nitid Studio: browser inference engine

This document describes the engine in `studio/src/engine/` and
`studio/src/workers/`, which runs an exported nitid ONNX model **in the browser**
with ONNX Runtime Web (ORT-web), inside a Web Worker. The image and the
computation never leave the page. Facts that were not verified are marked `[NV]`.

Scope: work package WP-2.3 (issue #44). The workbench UI that calls the engine
belongs to another work package; nothing in `src/` imports the engine yet.

## Architecture

```
page (React)                         Web Worker (same-origin module)
-------------                        --------------------------------
InferenceClient  --postMessage-->    inference.worker.ts
  init / run / dispose                 InferenceHandler (queue, in order)
  (src/engine/client.ts)                 |-- loadModelBytes  (Cache API, sha256 key)
                                         |-- EngineSession   (src/engine/session.ts)
                                               |-- preprocessRgba  -> float32 [1,3,S,S]
                                               |-- onnxruntime-web (wasm | webgpu)
                                               |-- decodeOutputs   -> Detection[]
```

| File | Responsibility |
| --- | --- |
| `engine/meta.ts` | Hand-written validator of `nitid_meta` (mirrors `api-contract/onnx-meta.schema.json`), class names, output layout. |
| `engine/onnx-metadata.ts` | Reads `metadata_props` straight from the ONNX protobuf (ORT-web does not expose it); `stripValueInfo` workaround. |
| `engine/preprocess.ts` | RGBA -> float32 CHW tensor. Pillow-exact "squash" resize; optional canvas mode. |
| `engine/decode.ts` | ONNX outputs -> detections: filters, box rescale/clip, mask resize/threshold/crop. |
| `engine/model-cache.ts` | Downloads the ONNX with a size cap, verifies sha256, caches it in the Cache API. |
| `engine/session.ts` | ORT setup (wasm paths, threads, providers), one inference at a time, limits. |
| `engine/client.ts` | Promise API over the worker protocol; `createInferenceClient()` starts the real worker. |
| `workers/protocol.ts` | Message types (`init`, `run`, `dispose`; `ok`, `error`, `progress`). |
| `workers/inference-handler.ts` | Request handling, separated from `self.onmessage` so it is testable. |
| `workers/inference.worker.ts`, `workers/ort-loader.ts` | The worker entry point and the dynamic import of ORT-web. |
| `engine/errors.ts` | `EngineError` with a stable `code`; serialisable across `postMessage`. |

Typical use:

```ts
const engine = createInferenceClient();
await engine.init({ modelUrl: "/api/v1/onnx/nitid1s?task=detect", sha256 }, onProgress);
const { detections, timings } = await engine.run({ kind: "bitmap", bitmap }, { conf: 0.5 });
await engine.dispose();
```

`init` takes either `modelUrl` (same-origin, cached) or `modelBytes` (a file the
user picked, transferred). `meta` is optional: if absent it is read from the
`nitid_meta` entry embedded in the ONNX (see `dfine/export_meta.py`).

## Contract with the model

The meta (`api-contract/onnx-meta.schema.json`, `schema_version` 1) promises:

- Input `images`: static float32 `[1, 3, S, S]`, RGB, range 0-1, **resize "squash"**.
- Post-processed outputs: `labels` int64 `[1,K]`, `boxes` float32 `[1,K,4]` as `xyxy` in
  pixels of the **model input** (S x S), `scores` float32 `[1,K]`, and for `segment`
  `masks` float32 `[1,K,S/4,S/4]`.
- Raw outputs (`export(postprocess=False)`): `pred_logits`, `pred_boxes` (normalised
  `cxcywh`), `pred_masks`. Both layouts are decoded.

If the ONNX does not expose the input/outputs the meta names, `EngineSession.create`
fails with `unsupported_output` and releases the session.

## What Python does, and what the engine replicates

Verified in the Python code (not taken from the task description):

| Topic | Python | Engine |
| --- | --- | --- |
| Preprocess | BGR frame -> RGB -> PIL -> `T.Resize((S,S))` -> `T.ToTensor()` (`dfine/utils/sources.py:330-341`). A pair size does not keep aspect ratio; PIL bilinear with antialiasing; alpha is dropped by `cv2.imread`. | Same, bit-exact (own port of Pillow's resampler). |
| Mask values | **Sigmoid probabilities**, not logits (`dfine/nn/architecture/decoder.py:1198-1199`, eval mode). | Treated as probabilities. |
| Mask threshold | `masks >= mask_threshold`, default **0.5** (`dfine/model.py:122`, `dfine/predictor.py:482`). | Same (`>=`). |
| Mask rescale | Low-res masks are interpolated straight to the original `(H, W)`: `F.interpolate(mode="bilinear", align_corners=False)` (`predictor.py:475-481`), then cropped to the clipped box, pixel `x` inside when `x1 <= x < x2` (`dfine/utils/ops.py:11-36`). | Same formula, computed only inside the box region; the mask is stored cropped (`DetectionMask`). |
| `conf` | `scores > conf` (strict), default **0.5**, not 0.25 (`predictor.py:428`, `model.py:121`). | Same. |
| `classes` | `torch.isin(labels, classes)` (`predictor.py:433-443`). | Same; `[]` keeps nothing. |
| Top-K | In the graph: top-K over the flattened `[queries x classes]` sigmoid scores (`dfine/nn/postprocessor.py:51-59`), `K = min(300, queries*classes)`. Exposed as `meta.topk`. | Not repeated for post-processed outputs; replayed for the raw layout. |
| Box scale | Normalised `cxcywh` -> `xyxy` -> `* (W, H)` (`postprocessor.py:47-48`); the ONNX deploy wrapper uses `(S, S)` as target. | `x * W / S`, `y * H / S` per axis. Equal up to float rounding. |
| Clipping | `clip_boxes` after the filters (`dfine/utils/ops.py:59-64`, `predictor.py:472`). | Same order. A box fully outside becomes a degenerate box and is kept, as in Python. |
| **NMS / `iou`** | **None in the normal path.** The only suppression is the cross-view merge of `augment=True` (flip TTA, `predictor.py:446-470`). `default_iou` (0.85) in the meta is informational. | **No NMS.** Overlapping same-class boxes are all kept. `augment` (TTA) is not implemented in v1. |
| Output order | Top-K order (descending score). | Preserved. |

Differences with the task description: it listed NMS among the things to
replicate "if Python does it". Python does not in the default path, so the engine
does not either (a test pins this). The default `conf` is 0.5.

`[NV]` Raw layout assumes `use_focal_loss=True` (sigmoid), the nitid default; the
meta does not record the flag and the softmax variant
(`postprocessor.py:60-76`) is not implemented.

`name` for an id missing from `meta.names` falls back to the decimal id (Python
behaviour for that case was not checked, `[NV]`).

### Detection shape

Matches the `Detection` schema of `api-contract/openapi.json` (`box` in pixels of
the original image, `confidence`, `class_id`, `name`, `has_mask`) plus the mask:

```ts
{ box: {x1,y1,x2,y2}, confidence, class_id, name, has_mask, mask?: {x, y, width, height, data: Uint8Array} }
```

`mask.data` is 0/1, row-major, covering only the clipped box (its top-left pixel is
`(x, y)` of the original image). `maskToFull()` expands it. Python zeroes
everything outside the box, so nothing is lost.

## Resampling difference: Pillow vs canvas

Python resizes with Pillow's bilinear, which **widens the filter when shrinking**
(antialiasing) and rounds to 8 bits after each pass with 22-bit fixed-point
weights. A canvas `drawImage` or a GPU bilinear sampler does not do the same.

- Default mode `"pil"`: `preprocess.ts` re-implements Pillow's algorithm
  (`Resample.c`: `precompute_coeffs`, `normalize_coeffs_8bpc`, horizontal then
  vertical pass). Result: **0 levels of difference** against Pillow 12.1.1 on four
  image sizes (see "Tolerances").
- Mode `"canvas"` (`run(image, { resize: "canvas" })`, `ImageBitmap` only): the browser scales
  with `imageSmoothingQuality = "high"`. Faster, **not** equivalent. Do not use it
  when results must match Python.
- Upscaling (the usual case for small inputs) differs by at most 1 level between
  any bilinear implementation; shrinking is where they drift.

An `ImageBitmap` in `"pil"` mode is first read back at its native size through an
`OffscreenCanvas` (`getImageData`) and then resized in JS. Canvas pixels are
premultiplied-alpha internally, so semi-transparent pixels lose precision; Python
drops alpha instead of compositing. For opaque images (photos) this is exact.

## Tolerances

Required by WP-2.3 and measured (fixtures from `make_parity_fixtures.py`, tiny
random model, nitid 0.9.x, torch 2.13, onnxruntime 1.28, Pillow 12.1.1):

| Quantity | Required | Measured |
| --- | --- | --- |
| Preprocess tensor vs Pillow/torchvision (8-bit levels, 4 images: 40x30, 150x110, 64x64, 97x33 -> 64x64; 48 samples + per-channel sums each) | documented | **0** (bit-exact) |
| Box IoU (TS vs `NITID.predict`), detect and segment, post-processed and raw layout, 4-5 cases each | >= 0.98 | >= 0.99999 |
| \|Delta score\| | <= 0.01 | <= 1.6e-6 |
| Mask IoU (segment) | >= 0.95 | 1.0 (all masks) |
| Canvas mode vs Pillow, headless Chrome, synthetic high-frequency image 640x480 -> 64 | documented | mean 2.9 levels, max 35 |
| Same, 150x110 -> 64 | documented | mean 1.2 levels, max 48 |
| Plain non-antialiased bilinear (the test reference model) vs Pillow, 640x480 -> 64 | documented | mean 10.6, max 122 |

The canvas figures come from a manual smoke run (below), not from the committed
tests; the synthetic image is a worst case (the blue channel is `(7x+13y) mod 256`),
natural photos drift less. Parity tests compare detections by class and box IoU,
not by row index, because the untrained tiny model gives all queries of one class
the same score.

Limits of the fixture model: random weights, 2 classes, 10 queries; scores are
class-dependent only (about 0.8 and 0.4 for the two classes). It checks the
decoder arithmetic and the filters; it says nothing about accuracy of a trained model.

## Limits and errors

All errors are `EngineError` with a `code`:

| Code | When |
| --- | --- |
| `invalid_meta` | meta missing, malformed or against the schema (message lists every problem). |
| `invalid_model` | the ONNX bytes are truncated or not a protobuf. |
| `invalid_image` | non-positive/fractional size, RGBA buffer of the wrong length. |
| `image_too_large` | more than **20 Mpx** (same limit as the REST API). |
| `model_too_large` | ONNX above 512 MiB (checked from `Content-Length` and while streaming). |
| `mask_budget_exceeded` | all masks of one result need more than 256 MiB; the message says to raise `conf` or shrink the image. |
| `model_fetch_failed`, `model_hash_mismatch` | download failed / bytes do not match the expected sha256. |
| `backend_unavailable` | ORT-web could not start, no `OffscreenCanvas`/`crypto.subtle`. |
| `unsupported_output` | ONNX outputs do not match the meta, or tensor dims are not batch 1. |
| `inference_failed`, `session_not_ready`, `invalid_options`, `worker_error` | ORT run error; `run` before `init`/after `dispose`; bad option; worker crash. |

The 25 MB file limit of the REST API is for uploads and is enforced by the
frontend that reads the file, not by the engine (it receives pixels).

## Model cache

`model-cache.ts` stores the ONNX in the Cache API (`nitid-studio-onnx-v1`) under
`<origin>/__nitid-onnx__/<sha256>`:

- The key is the meta `sha256` or, failing that, a strong ETag `"<hex64>"` obtained with a
  `HEAD` request (`api-contract/README.md`, "ONNX ETag"). Content addressing: a new model
  version gets a new key; an entry can never be stale.
- Without any known key the file is downloaded every time and **not** cached.
- Hits and downloads are hashed (`crypto.subtle`) and compared with the key; a corrupt entry is deleted
  and re-downloaded; a mismatching download is `model_hash_mismatch`.
- At most 4 models are kept (oldest evicted). If the cache cannot be written (quota, private
  mode) loading still works.
- Measured once in headless Chrome (35 MB segment model, 1 thread): first load 1.3-1.6 s including the
  download from localhost, second load from the cache 0.7-1.2 s (hashing and session creation dominate).
  These are indications, not benchmarks.

## Running ORT-web: files, CSP, threads, WebGPU

- **No CDN.** Everything is same-origin. Target CSP:
  `script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:` (verified in headless Chrome
  with exactly that header, including a cross-origin-isolated page with 2 threads; `blob:` is
  needed by ORT's thread workers).
- **Runtime files.** `scripts/copy-ort.mjs` (hooked as `predev`/`prebuild`/`prebuild:hosted`) copies
  `ort-wasm-simd-threaded.{mjs,wasm}` (15 MB) from `node_modules/onnxruntime-web/dist` to
  `public/ort/` (git-ignored, never commit `.wasm`; `pnpm run check:size` forbids it). Vite copies
  `public/` to the build output, so **the 15 MB wasm ends up in `nitid/studio/static`** and in the
  Python wheel unless packaging excludes it.
- **Extern wasm entry points.** `vite.config.ts` adds the resolve condition
  `onnxruntime-web-use-extern-wasm`; without it Vite would embed 15 MB and 29 MB of wasm in the
  JS assets. With it the bundle only contains ORT's small JS and the wasm is loaded from `/ort/`.
- **`env.wasm.wasmPaths` is absolute** (`/ort/`). A relative prefix is resolved against the script URL
  and ends up duplicated. `assertAbsoluteWasmBase` rejects relative values.
- **Serving.** `.wasm` must be `application/wasm` and `.mjs` a JavaScript MIME type (Python's
  `mimetypes` knows both). `[NV]` the Python server (other work package) has to do this.
- **Threads.** `numThreads = crossOriginIsolated ? min(4, cores) : 1`. Multi-threading needs
  `SharedArrayBuffer`, i.e. `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` on the page. `[NV]` whether the local server will
  send them; without them the engine is correct but single-threaded.
- **WebGPU.** Optional: `backend: "webgpu" | "auto"` loads the separate `onnxruntime-web/webgpu`
  build only if `navigator.gpu.requestAdapter()` returns an adapter, and falls back to wasm on
  any error (`info.fallbackReason`). The WebGPU build needs `ort-wasm-simd-threaded.jsep.{mjs,wasm}`
  (29 MB more), copied only with `ORT_COPY_WEBGPU=1` (or `--webgpu`). `[NV]` **WebGPU was never
  run on a real GPU here**: headless Chrome exposes `navigator.gpu` but returns no adapter, so
  only the fallback path was exercised.

## Known problem: nitid exports that ORT-web 1.31.1 rejects

The `segment` ONNX exported by the current exporter fails to load in ORT-web 1.31.1 with
`ShapeInferenceError ... Mismatch between number of inferred and declared dimensions. inferred=2
declared=0`. Cause (checked with `onnx`): the graph has a `value_info` entry `_v_2148` declared with
rank 0 for a `Gemm` output that is `[4, 256]`. Python `onnxruntime` 1.28 accepts it with a warning
("Falling back to lenient merge"). The `detect` export loaded fine.

- Engine workaround (committed, tested): on that error the session retries once with the bytes
  minus `graph.value_info` (`stripValueInfo`, 25 ms for 35 MB; computation unchanged). The result is
  reported in `info.workarounds = ["stripped-value-info"]`.
- Proper fix (outside this work package, `dfine/exporter.py`): clear `graph.value_info` (or fix that entry)
  after export. Then the workaround never triggers.

## Regenerating the parity fixtures

```bash
cd <repo>
PYTHONPATH=<repo> <venv>/bin/python studio/tools/make_parity_fixtures.py
```

It builds a tiny checkpoint at run time (no weights are downloaded), exports ONNX in both
layouts, runs PyTorch and ONNX Runtime, and writes
`studio/src/engine/__fixtures__/parity.{detect,segment}.json` (about 88 KB in total; it exits
with an error above 100 KB). The synthetic image is a formula (`gradient-v1`) re-implemented in
`studio/src/engine/test-utils.ts`; keep both in sync. No `.onnx` or image is ever stored.
Regenerate when the exporter, the post-processor or the predictor change, and review the diff.

## Tests

`pnpm run test` (from `studio/`). Engine and worker tests use no WebAssembly: ORT-web is replaced by a
fake session that replays the fixture outputs. They cover Pillow parity, decoding parity,
filters, strictness of `conf`, no NMS, clipping, masks (threshold, interpolation, crop, budget),
raw layout, meta validation against every `meta.*.json` contract fixture, the ONNX protobuf reader,
the cache (hits, keys, corruption, eviction, quota, size limits), the session (providers,
threads, fallback, serialisation, limits, errors) and the worker protocol (ordering, transfer,
crash, terminate).

A real-browser smoke test (headless Chrome via `playwright-core` driving a Vite build under the target CSP, a
generated tiny ONNX in `/tmp`) was run manually and is **not committed**. It confirmed: same-origin wasm
loading, no CDN request, cache hit on reload, 1 thread without isolation and 2 with it,
detections equal to the fixtures (box IoU 0.999997, |Delta score| 5e-7), and the `value_info` workaround.
Timings on the 2-CPU CI-like machine, 64x64 model, 1 thread, headless, no GPU: inference
about 20 ms (detect) and 40-45 ms (segment) per image after a warm-up first run of about 270 ms. These
say nothing about a 640x640 model or a real GPU.

## Limitations

- `augment` (flip TTA) and the `iou` argument are not implemented.
- Raw layout: sigmoid only (`[NV]` softmax variant).
- Semantic segmentation exports are not described by the meta and not supported.
- Batch size 1 only (the meta shape is static `[1,3,S,S]`).
- `[NV]` WebGPU and multi-threading in a real browser.
- Large photos: Pillow-exact resizing runs in JS in the worker (about 0.15 s for 12 Mpx, estimated, not measured).

## Dependencies

`onnxruntime-web` 1.31.1, licence **MIT**, added as a runtime dependency (the `.wasm`
files it ships are part of the same package). Its transitive packages, with the licence read from
each installed `package.json`: `onnxruntime-common` (MIT), `flatbuffers` (Apache-2.0),
`guid-typescript` (ISC), `long` (Apache-2.0), `platform` (MIT), `protobufjs` and the
`@protobufjs/*` helpers (BSD-3-Clause). All are permissive. `THIRD_PARTY_NOTICES.md` was not
edited here (outside this work package) and should list them if the wasm is shipped.
No `ajv` (it needs `'unsafe-eval'`). Python side: nothing new (uses nitid, torch, onnx,
onnxruntime already in the dev environment).
