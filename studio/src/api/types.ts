/**
 * Minimal hand-written API types.
 *
 * TODO: these will be generated from `studio/api-contract/openapi.json` once
 * that contract exists (separate work package). Until then this file is the
 * single place that describes what the UI expects from the backend. Anything
 * marked [NV] is an assumption that has not been verified against the server.
 */

/** API version this UI build was written against. [NV] value and format. */
export const EXPECTED_API_VERSION = "1";

/** Body of `GET /api/v1/health` (validated at runtime by `parseHealth`). */
export interface HealthResponse {
  status: "ok";
  nitid_version: string;
  /** Normalised to a string even if the server sends a number. [NV] */
  api_version: string;
}

/** Common error envelope; the real shape is [NV] until the contract exists. */
export interface ApiErrorBody {
  detail?: string;
  message?: string;
}

/** Where the model runs. */
export type RunMode = "local" | "browser";

/** Which deployment of the SPA this is. */
export type BackendMode = "local" | "hosted";
