/**
 * Loads the real `onnxruntime-web` build. Two entry points are used so that the
 * WebGPU build (and its 29 MB `jsep` wasm) is only fetched when asked for:
 *
 * - `onnxruntime-web/wasm`   -> `ort-wasm-simd-threaded.{mjs,wasm}`
 * - `onnxruntime-web/webgpu` -> `ort-wasm-simd-threaded.jsep.{mjs,wasm}`
 *
 * Both pairs are copied to `public/ort/` by `scripts/copy-ort.mjs`.
 */
import type { OrtModule } from "../engine/session";

export async function loadOrt(provider: "wasm" | "webgpu"): Promise<OrtModule> {
  const module =
    provider === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");
  return module as unknown as OrtModule;
}
