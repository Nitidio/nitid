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
  expectedApiVersion?: number;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

/** Health probe timeout: a static host answers (or refuses) fast. */
export const HEALTH_TIMEOUT_MS = 2_000;

/**
 * The contract says integer. Servers that predate it sent a string (`"1"`), so a string of
 * digits is still accepted and converted; anything else is not a health body we understand.
 */
function parseApiVersion(value: unknown): number | null {
  if (typeof value === "number") return Number.isInteger(value) && value >= 0 ? value : null;
  if (typeof value === "string" && /^\d{1,6}$/.test(value.trim())) return Number(value.trim());
  return null;
}

/** Validate the `/health` body; returns null if it is not what the contract promises. */
export function parseHealth(value: unknown): HealthResponse | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (v.status !== "ok") return null;
  if (typeof v.nitid_version !== "string" || !v.nitid_version) return null;
  const api = parseApiVersion(v.api_version);
  if (api === null) return null;
  return { status: "ok", nitid_version: v.nitid_version, api_version: api };
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
