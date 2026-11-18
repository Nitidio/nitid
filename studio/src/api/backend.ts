import { createApiClient } from "./client";
import { EXPECTED_API_VERSION, type BackendMode, type HealthResponse } from "./types";

export interface BackendInfo {
  mode: BackendMode;
  /** Present only in "local" mode. */
  health?: HealthResponse;
  /** True when the backend reports an `api_version` this UI does not expect. */
  versionMismatch: boolean;
}

export interface DetectOptions {
  /** Defaults to the build-time `__STUDIO_MODE__`. "hosted" skips probing entirely. */
  studioMode?: "auto" | "hosted";
  expectedApiVersion?: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

/** Health probe timeout: a static host answers (or refuses) fast. */
export const HEALTH_TIMEOUT_MS = 2_000;

/** Validate the `/health` body; returns null if it is not what the contract promises. */
export function parseHealth(value: unknown): HealthResponse | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (v.status !== "ok") return null;
  if (typeof v.nitid_version !== "string" || !v.nitid_version) return null;
  const api = v.api_version;
  if (typeof api !== "string" && typeof api !== "number") return null;
  return { status: "ok", nitid_version: v.nitid_version, api_version: String(api) };
}

/**
 * Decide between "local" (backend present) and "hosted" (static, browser only).
 *
 * Never throws and never logs: any failure (404, HTML fallback page, network
 * error, timeout, bad JSON) means "hosted". The health probe is deliberately
 * sent without the bearer token: it is a liveness check, and a static host has
 * no business receiving the token.
 */
export async function detectBackend(options: DetectOptions = {}): Promise<BackendInfo> {
  const studioMode = options.studioMode ?? __STUDIO_MODE__;
  if (studioMode === "hosted") return { mode: "hosted", versionMismatch: false };

  const client = createApiClient({
    getToken: () => null,
    timeoutMs: options.timeoutMs ?? HEALTH_TIMEOUT_MS,
    ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
  });

  try {
    const health = parseHealth(await client.request<unknown>("/health"));
    if (!health) return { mode: "hosted", versionMismatch: false };
    const expected = options.expectedApiVersion ?? EXPECTED_API_VERSION;
    return { mode: "local", health, versionMismatch: health.api_version !== expected };
  } catch {
    return { mode: "hosted", versionMismatch: false };
  }
}
