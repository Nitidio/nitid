/**
 * Loading of the ONNX bytes with a content-addressed Cache API cache.
 *
 * The cache key is the lowercase sha256 of the file. It comes from the meta
 * (`sha256`, added by the REST API) or from the strong `ETag` of the response
 * (`"<hex64>"`, `api-contract/README.md`, "ONNX ETag"). Content addressing means
 * a cached entry can never be stale: a changed model has a different key.
 *
 * - No known key (for example a user file served without a sha256 ETag): the
 *   file is downloaded every time and NOT cached. We never invent a weak key.
 * - On every hit and every download the bytes are hashed and compared with the
 *   key; a mismatch is an error (`model_hash_mismatch`) and a corrupt cache
 *   entry is deleted and re-downloaded.
 * - At most `maxEntries` models are kept (oldest inserted first are evicted).
 * - The model comes from the local server (same origin) or, in the hosted deployment, from the
 *   configured models origin (CORS, no credentials; `requestInit`). The cache key is built from the
 *   page's own origin, never from the model's. If the cache cannot be written (quota, private mode)
 *   loading still works and `cached` is `false`.
 * - Across origins, the `ETag` / `Content-Length` of a response are only readable when the server sends
 *   `Access-Control-Expose-Headers: ETag, Content-Length`. Without them: no HEAD key (the catalogue
 *   `sha256` is then the only key) and no progress total; the streamed size cap still applies.
 */
import { EngineError } from "./errors";
import { MAX_MODEL_BYTES } from "./types";

export const MODEL_CACHE_NAME = "nitid-studio-onnx-v1";
const SHA256_RE = /^[0-9a-f]{64}$/;

export interface CacheLike {
  match(request: string): Promise<Response | undefined>;
  put(request: string, response: Response): Promise<void>;
  delete(request: string): Promise<boolean>;
  keys(): Promise<readonly { url: string }[]>;
}
export interface CacheStorageLike {
  open(name: string): Promise<CacheLike>;
}

export interface LoadModelOptions {
  url: string;
  /** Known sha256 (from the meta), lowercase hex. */
  sha256?: string;
  /**
   * Extra request headers for the HEAD and GET (for example `Authorization: Bearer ...` for the
   * local server's `/api/v1/onnx/{model}`). Never stored in the cache.
   */
  headers?: Readonly<Record<string, string>>;
  fetchFn?: typeof fetch;
  caches?: CacheStorageLike | null;
  cacheName?: string;
  /** Origin used to build synthetic cache keys; defaults to `self.location.origin`. */
  origin?: string;
  maxBytes?: number;
  maxEntries?: number;
  onProgress?: (loaded: number, total: number | undefined) => void;
  /** Injectable for tests; defaults to `crypto.subtle.digest("SHA-256")`. */
  sha256Hex?: (bytes: ArrayBuffer) => Promise<string>;
}

export interface LoadedModel {
  bytes: ArrayBuffer;
  source: "cache" | "network";
  sha256: string | undefined;
  /** True when this call wrote the file to the cache. */
  cached: boolean;
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new EngineError("backend_unavailable", "crypto.subtle is not available (a secure context is required)");
  }
  const digest = new Uint8Array(await subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Extracts the sha256 from a strong ETag `"<hex64>"`; `null` for anything else (weak, other formats). */
export function sha256FromEtag(etag: string | null): string | null {
  if (!etag) return null;
  const match = /^"([0-9a-f]{64})"$/.exec(etag.trim());
  return match ? match[1]! : null;
}

function cacheKey(origin: string, sha256: string): string {
  return `${origin}/__nitid-onnx__/${sha256}`;
}

export async function loadModelBytes(options: LoadModelOptions): Promise<LoadedModel> {
  const fetchFn = options.fetchFn ?? fetch.bind(globalThis);
  const hash = options.sha256Hex ?? sha256Hex;
  const maxBytes = options.maxBytes ?? MAX_MODEL_BYTES;
  const maxEntries = options.maxEntries ?? 4;
  // Cache keys must be absolute same-origin URLs. Without a known origin (no `location`) we do not cache.
  const origin = options.origin ?? globalThis.location?.origin;
  const cacheStorage = origin === undefined ? null : options.caches === undefined ? (globalThis.caches ?? null) : options.caches;

  if (options.sha256 !== undefined && !SHA256_RE.test(options.sha256)) {
    throw new EngineError("invalid_options", "sha256 must be 64 lowercase hex characters");
  }

  let cache: CacheLike | null = null;
  if (cacheStorage) {
    try {
      cache = await cacheStorage.open(options.cacheName ?? MODEL_CACHE_NAME);
    } catch {
      cache = null; // Cache API unusable (e.g. private mode): carry on without it.
    }
  }

  // 1. Find the key without downloading the body (meta sha256, else HEAD + ETag).
  let key = options.sha256;
  if (!key && cache) {
    try {
      const head = await fetchFn(options.url, { method: "HEAD", ...requestInit(options.headers) });
      if (head.ok) key = sha256FromEtag(head.headers.get("etag")) ?? undefined;
    } catch {
      // HEAD unsupported or network error: fall through to the plain download.
    }
  }

  // 2. Cache hit (verified).
  if (key && cache) {
    const entry = cacheKey(origin, key);
    const hit = await cache.match(entry).catch(() => undefined);
    if (hit) {
      const bytes = await hit.arrayBuffer();
      if (bytes.byteLength <= maxBytes && (await hash(bytes)) === key) {
        options.onProgress?.(bytes.byteLength, bytes.byteLength);
        return { bytes, source: "cache", sha256: key, cached: false };
      }
      await cache.delete(entry).catch(() => false); // corrupt entry: drop it and download again
    }
  }

  // 3. Download with a size cap.
  let response: Response;
  try {
    response = await fetchFn(options.url, requestInit(options.headers));
  } catch (error) {
    throw new EngineError("model_fetch_failed", `Could not download the model: ${String(error)}`, { cause: error });
  }
  if (!response.ok) {
    throw new EngineError("model_fetch_failed", `Could not download the model: HTTP ${response.status}`);
  }
  const lengthHeader = Number(response.headers.get("content-length"));
  const total = Number.isFinite(lengthHeader) && lengthHeader > 0 ? lengthHeader : undefined;
  if (total !== undefined && total > maxBytes) throw tooLarge(total, maxBytes);
  const bytes = await readBody(response, maxBytes, total, options.onProgress);

  // 4. Verify against the best-known key and cache.
  const digest = await hash(bytes);
  const etagKey = sha256FromEtag(response.headers.get("etag"));
  const expected = key ?? etagKey ?? undefined;
  if (expected && digest !== expected) {
    throw new EngineError(
      "model_hash_mismatch",
      `The downloaded model has sha256 ${digest} but ${expected} was expected. The file is corrupt or was replaced.`,
    );
  }

  let cached = false;
  if (expected && cache) {
    try {
      await cache.put(
        cacheKey(origin, expected),
        new Response(bytes, { headers: { "content-type": "application/octet-stream" } }),
      );
      await prune(cache, maxEntries);
      cached = true;
    } catch {
      cached = false; // quota exceeded or storage disabled: not fatal
    }
  }
  return { bytes, source: "network", sha256: expected ?? digest, cached };
}

/**
 * Options of every model request. They are explicit because the model may live on another origin
 * (hosted deployment, `docs/studio-hosted.md`): the page is cross-origin isolated (COEP
 * `require-corp`), so a cross-origin response is only usable when it passes CORS, hence
 * `mode: "cors"`; and the request must never carry cookies or other ambient credentials, hence
 * `credentials: "omit"` (the local server authenticates with the explicit `Authorization` header).
 * A header list is added only when given: without custom headers the request stays a CORS "simple"
 * request and needs no preflight.
 */
export function requestInit(headers: Readonly<Record<string, string>> | undefined): RequestInit {
  const init: RequestInit = { mode: "cors", credentials: "omit" };
  if (headers && Object.keys(headers).length > 0) init.headers = { ...headers };
  return init;
}

function tooLarge(size: number, limit: number): EngineError {
  return new EngineError(
    "model_too_large",
    `The model is ${(size / 2 ** 20).toFixed(0)} MiB; the limit is ${(limit / 2 ** 20).toFixed(0)} MiB.`,
  );
}

async function readBody(
  response: Response,
  maxBytes: number,
  total: number | undefined,
  onProgress: LoadModelOptions["onProgress"],
): Promise<ArrayBuffer> {
  if (!response.body) {
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > maxBytes) throw tooLarge(bytes.byteLength, maxBytes);
    onProgress?.(bytes.byteLength, total);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    loaded += value.byteLength;
    if (loaded > maxBytes) {
      await reader.cancel().catch(() => undefined);
      throw tooLarge(loaded, maxBytes);
    }
    chunks.push(value);
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

async function prune(cache: CacheLike, maxEntries: number): Promise<void> {
  const keys = (await cache.keys()).filter((request) => request.url.includes("/__nitid-onnx__/"));
  // Cache.keys() returns entries in insertion order: evict the oldest.
  for (const request of keys.slice(0, Math.max(0, keys.length - maxEntries))) {
    await cache.delete(request.url);
  }
}
