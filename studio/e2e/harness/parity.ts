/**
 * Parity harness: runs the real browser engine (src/engine + the inference worker + ORT-web) on an
 * image, with the ONNX downloaded from the real `GET /api/v1/onnx/{model}` endpoint, and returns
 * detections in a JSON-friendly shape. Driven by studio/e2e/parity.mjs.
 */
import { createInferenceClient } from "../../src/engine";
import type { Backend, Detection } from "../../src/engine";

interface RunArgs {
  token: string;
  model: string;
  task: "detect" | "segment";
  imageUrl: string;
  confs: number[];
  backend?: Backend;
  /** Let the worker download the model itself (no Authorization header possible). */
  viaWorkerUrl?: boolean;
}

function toBase64(bytes: Uint8Array): string {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

function serialise(detections: Detection[]) {
  return detections.map((d) => ({
    box: d.box,
    confidence: d.confidence,
    class_id: d.class_id,
    name: d.name,
    has_mask: d.has_mask,
    mask: d.mask
      ? { x: d.mask.x, y: d.mask.y, width: d.mask.width, height: d.mask.height, data: toBase64(d.mask.data) }
      : null,
  }));
}

async function environment() {
  const nav = navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } };
  let adapter: string = "navigator.gpu missing";
  if (nav.gpu) {
    try {
      adapter = (await nav.gpu.requestAdapter()) === null ? "requestAdapter() returned null" : "adapter available";
    } catch (error) {
      adapter = `requestAdapter() threw: ${String(error)}`;
    }
  }
  return {
    crossOriginIsolated: window.crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer,
    hardwareConcurrency: navigator.hardwareConcurrency,
    webgpu: adapter,
    userAgent: navigator.userAgent,
  };
}

async function run(args: RunArgs) {
  const engine = createInferenceClient();
  try {
    let modelInit: Parameters<typeof engine.init>[0];
    const url = `/api/v1/onnx/${encodeURIComponent(args.model)}?task=${args.task}`;
    let etag: string | null = null;
    if (args.viaWorkerUrl) {
      modelInit = { modelUrl: url, backend: args.backend ?? "wasm" };
    } else {
      const response = await fetch(url, { headers: { Authorization: `Bearer ${args.token}` } });
      if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
      etag = response.headers.get("etag");
      const sha256 = etag?.replace(/"/g, "") ?? undefined;
      modelInit = { modelBytes: await response.arrayBuffer(), ...(sha256 ? { sha256 } : {}), backend: args.backend ?? "wasm" };
    }
    const t0 = performance.now();
    const info = await engine.init(modelInit);
    const initMs = performance.now() - t0;

    const imageBlob = await (await fetch(args.imageUrl)).blob();
    const results: Record<string, unknown> = {};
    const timings: number[] = [];
    for (const conf of args.confs) {
      const bitmap = await createImageBitmap(imageBlob);
      const t1 = performance.now();
      const out = await engine.run({ kind: "bitmap", bitmap }, { conf });
      timings.push(performance.now() - t1);
      results[String(conf)] = {
        detections: serialise(out.detections),
        width: out.imageSize.width,
        height: out.imageSize.height,
        timings: out.timings,
      };
    }
    // The same image again: the engine must be deterministic and survive repeated runs.
    const again = await engine.run({ kind: "bitmap", bitmap: await createImageBitmap(imageBlob) }, { conf: args.confs[0] ?? 0.5 });
    return { etag, info, initMs, runMs: timings, results, repeatCount: again.detections.length };
  } finally {
    engine.terminate();
  }
}

declare global {
  interface Window {
    __parity: { environment: typeof environment; run: typeof run };
  }
}
window.__parity = { environment, run };
