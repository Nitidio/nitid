/**
 * Typed errors of the browser inference engine.
 *
 * Every failure that can reach the UI is an {@link EngineError} with a stable,
 * machine-readable `code` (snake_case, same style as the REST API error
 * envelope in `api-contract/openapi.json`) and a human-readable message.
 */
export type EngineErrorCode =
  | "invalid_meta"
  | "invalid_model"
  | "invalid_image"
  | "invalid_options"
  | "image_too_large"
  | "model_too_large"
  | "mask_budget_exceeded"
  | "model_fetch_failed"
  | "model_hash_mismatch"
  | "backend_unavailable"
  | "session_not_ready"
  | "unsupported_output"
  | "inference_failed"
  | "worker_error";

export class EngineError extends Error {
  readonly code: EngineErrorCode;

  constructor(code: EngineErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "EngineError";
    this.code = code;
  }
}

/** Plain-object form that survives `postMessage` (Error subclasses do not). */
export interface SerializedEngineError {
  code: EngineErrorCode;
  message: string;
}

export function serializeError(error: unknown): SerializedEngineError {
  if (error instanceof EngineError) return { code: error.code, message: error.message };
  const message = error instanceof Error ? error.message : String(error);
  return { code: "worker_error", message };
}

export function deserializeError(error: SerializedEngineError): EngineError {
  return new EngineError(error.code, error.message);
}
