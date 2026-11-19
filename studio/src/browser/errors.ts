import { EngineError, type EngineErrorCode } from "../engine/errors";
import { describeError, type DescribedError } from "../local/errors";
import { CatalogError } from "./catalog";

const ENGINE_HINTS: Partial<Record<EngineErrorCode, string>> = {
  image_too_large: "Use an image below 20 megapixels (the same limit as the server).",
  model_too_large: "This model is too large for the browser. Use a smaller size, or server mode.",
  model_fetch_failed: "The model could not be downloaded. Check that the server or the hosting is reachable and try again.",
  model_hash_mismatch: "The downloaded file does not match its checksum. Reload the page; if it persists, the model file changed or is damaged.",
  backend_unavailable: "This browser cannot run the model (WebAssembly or secure-context features are missing). Try a current Chrome, Edge or Firefox.",
  mask_budget_exceeded: "Raise the confidence or use a smaller image: too many mask pixels for the browser to hold.",
  unsupported_output: "This ONNX file does not look like a Nitid export.",
  invalid_meta: "This ONNX file does not carry a valid Nitid descriptor.",
  invalid_model: "The file is not a valid ONNX model.",
};

/** Engine, catalogue and server errors as text for the panel. Never includes the token. */
export function describeBrowserError(err: unknown): DescribedError {
  if (err instanceof EngineError) {
    const hint = ENGINE_HINTS[err.code];
    return { message: err.message, code: err.code, ...(hint ? { hint } : {}) };
  }
  if (err instanceof CatalogError) return { message: err.message };
  return describeError(err);
}
