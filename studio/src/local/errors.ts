import { ApiError } from "../api/client";
import { PollTimeoutError, RemoteFailure } from "./polling";

export interface DescribedError {
  message: string;
  /** Contract error code when the server sent one. */
  code?: string;
  /** What the user can do next. */
  hint?: string;
}

const HINTS: Record<string, string> = {
  unauthorized:
    "The session token is missing or expired. Open Studio from the link printed by `nitid studio`.",
  forbidden_origin:
    "The server only accepts requests from its own address. Open the exact URL printed by `nitid studio`.",
  payload_too_large: "Images are limited to 25 MB and 20 megapixels each. Remove or resize the large ones.",
  unsupported_media_type: "Only image files are accepted here (JPEG, PNG, WebP...).",
  invalid_params: "The server rejected the settings. Check the model, task and device selection.",
  not_found: "The server no longer has this item. Reload the model list and try again.",
  conflict: "The server is busy with this model already. Wait for it to finish, then retry.",
};

/** Turn any failure into text the panel can show. Never includes the token. */
export function describeError(err: unknown): DescribedError {
  if (err instanceof ApiError) {
    const code = err.code ?? (err.kind === "unauthorized" ? "unauthorized" : undefined);
    const hint =
      (code && HINTS[code]) ||
      (err.kind === "network"
        ? "Is `nitid studio` still running? Check the terminal where you started it."
        : err.kind === "timeout"
          ? "The server did not answer in time. It may be busy; try again."
          : undefined);
    return { message: err.message, ...(code ? { code } : {}), ...(hint ? { hint } : {}) };
  }
  if (err instanceof RemoteFailure || err instanceof PollTimeoutError) return { message: err.message };
  if (err instanceof Error) return { message: err.message };
  return { message: "Something went wrong." };
}

export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}
