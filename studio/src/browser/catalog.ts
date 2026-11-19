/**
 * Models the in-browser mode can load.
 *
 * - Local deployment: the models of `GET /models` that are downloaded; their ONNX comes from
 *   `GET /api/v1/onnx/{id}?task=` (bearer token) and its sha256 from `.../meta`.
 * - Hosted deployment: a catalogue `models.json` under the configured base URL
 *   (`config.ts`). Format (version 1):
 *
 *       { "models": [ { "id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx",
 *                       "sha256": "<64 hex, optional>", "label": "optional" } ] }
 *
 *   `url` is resolved against the base URL. `sha256` is the cache key and integrity check; without
 *   it the engine tries a HEAD request for a strong ETag and otherwise does not cache.
 */
import { API_PREFIX } from "../api/client";
import type { Model, Task } from "../api/types";
import { modelKey } from "../local/models";

export interface BrowserModel {
  /** `task:id`, unique in a list. */
  key: string;
  id: string;
  task: Task;
  label: string;
  origin: "server" | "hosted";
  /** Absolute path or URL the worker downloads. */
  url: string;
  sha256?: string;
}

const SHA256_RE = /^[0-9a-f]{64}$/;

/** `/api/v1/onnx/{id}?task=` (no `imgsz`: the server's default size). */
export function serverOnnxUrl(id: string, task: Task): string {
  return `${API_PREFIX}/onnx/${encodeURIComponent(id)}?task=${encodeURIComponent(task)}`;
}

export function serverMetaPath(id: string, task: Task): string {
  return `/onnx/${encodeURIComponent(id)}/meta?task=${encodeURIComponent(task)}`;
}

/** Models the server can export to ONNX right now: those it has weights for. */
export function fromServerModels(models: readonly Model[]): BrowserModel[] {
  return models
    .filter((m) => m.downloaded)
    .map((m) => ({
      key: modelKey(m),
      id: m.id,
      task: m.task,
      label: `${m.id} (${m.task})`,
      origin: "server" as const,
      url: serverOnnxUrl(m.id, m.task),
    }));
}

export class CatalogError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CatalogError";
  }
}

/** Validates a hosted `models.json`. Entries that are not understood are reported, not skipped silently. */
export function parseHostedCatalog(value: unknown, baseUrl: string): BrowserModel[] {
  const list = (value as { models?: unknown } | null)?.models;
  if (!Array.isArray(list)) throw new CatalogError('The model catalogue must be {"models": [...]}.');
  const out: BrowserModel[] = [];
  const seen = new Set<string>();
  list.forEach((raw, i) => {
    const e = raw as Record<string, unknown> | null;
    const where = `models[${i}]`;
    if (!e || typeof e !== "object") throw new CatalogError(`${where} is not an object.`);
    if (typeof e.id !== "string" || !e.id) throw new CatalogError(`${where}.id must be a non-empty string.`);
    if (e.task !== "detect" && e.task !== "segment") throw new CatalogError(`${where}.task must be "detect" or "segment".`);
    if (typeof e.url !== "string" || !e.url) throw new CatalogError(`${where}.url must be a non-empty string.`);
    if (e.sha256 !== undefined && (typeof e.sha256 !== "string" || !SHA256_RE.test(e.sha256))) {
      throw new CatalogError(`${where}.sha256 must be 64 lowercase hex characters.`);
    }
    if (e.label !== undefined && typeof e.label !== "string") throw new CatalogError(`${where}.label must be a string.`);
    const key = modelKey({ id: e.id, task: e.task });
    if (seen.has(key)) throw new CatalogError(`${where}: duplicate model ${key}.`);
    seen.add(key);
    out.push({
      key,
      id: e.id,
      task: e.task,
      label: typeof e.label === "string" && e.label ? e.label : `${e.id} (${e.task})`,
      origin: "hosted",
      url: new URL(e.url, new URL(baseUrl, globalThis.location?.href ?? "http://localhost/")).href,
      ...(typeof e.sha256 === "string" ? { sha256: e.sha256 } : {}),
    });
  });
  return out;
}

/** Fetches and validates `<base>/models.json`. Throws `CatalogError` with a user-facing message. */
export async function loadHostedCatalog(baseUrl: string, fetchImpl: typeof fetch = fetch): Promise<BrowserModel[]> {
  let response: Response;
  try {
    response = await fetchImpl(`${baseUrl}models.json`, { headers: { Accept: "application/json" } });
  } catch {
    throw new CatalogError("The model catalogue could not be reached.");
  }
  if (!response.ok) throw new CatalogError(`The model catalogue answered HTTP ${response.status}.`);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new CatalogError("The model catalogue is not valid JSON.");
  }
  return parseHostedCatalog(body, baseUrl);
}
