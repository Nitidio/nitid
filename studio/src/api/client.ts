import { getToken } from "../token";
import type { ApiErrorBody } from "./types";

export const API_PREFIX = "/api/v1";
export const DEFAULT_TIMEOUT_MS = 10_000;

export type ApiErrorKind =
  /** The request never got a response (connection refused, offline...). */
  | "network"
  /** The request exceeded the timeout and was aborted. */
  | "timeout"
  /** 401 or 403: missing, wrong or expired token. */
  | "unauthorized"
  /** Any other non-2xx status. */
  | "http"
  /** 2xx but the body is not valid JSON. */
  | "invalid-response";

/** Typed error thrown by every client call. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | undefined;
  /** snake_case code from the contract error envelope (`{"error":{"code",...}}`), when present. */
  readonly code: string | undefined;

  constructor(
    kind: ApiErrorKind,
    message: string,
    status?: number,
    options?: ErrorOptions & { code?: string },
  ) {
    super(message, options);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.code = options?.code;
  }
}

export interface ClientOptions {
  /** Defaults to same-origin (""). */
  baseUrl?: string;
  /** Token source; defaults to the in-memory token from `token.ts`. */
  getToken?: () => string | null;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export interface RequestOptions {
  method?: string;
  /** Serialised as JSON. Use `formData` for uploads. */
  json?: unknown;
  formData?: FormData;
  signal?: AbortSignal;
  /** Overrides the client timeout for this call (uploads need more than a health probe). */
  timeoutMs?: number;
  /** "blob" returns the body as a `Blob` (images) instead of parsing JSON. */
  responseType?: "json" | "blob";
}

export interface ApiClient {
  request<T>(path: string, options?: RequestOptions): Promise<T>;
}

/** Read the contract envelope `{"error":{"code","message"}}`, falling back to `detail`/`message`. */
export function parseErrorBody(status: number, body: unknown): { code?: string; message: string } {
  if (body && typeof body === "object") {
    const { error, detail, message } = body as ApiErrorBody;
    if (error && typeof error === "object") {
      const code = typeof error.code === "string" && error.code ? error.code : undefined;
      if (typeof error.message === "string" && error.message) {
        return { ...(code ? { code } : {}), message: error.message };
      }
      if (code) return { code, message: `Request failed with status ${status}` };
    }
    if (typeof detail === "string" && detail) return { message: detail };
    if (typeof message === "string" && message) return { message };
  }
  return { message: `Request failed with status ${status}` };
}

/** Same-origin JSON client that adds the bearer token and maps failures to `ApiError`. */
export function createApiClient(options: ClientOptions = {}): ApiClient {
  const baseUrl = options.baseUrl ?? "";
  const tokenSource = options.getToken ?? getToken;
  const defaultTimeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  async function request<T>(path: string, req: RequestOptions = {}): Promise<T> {
    const doFetch = options.fetchImpl ?? fetch;
    const timeoutMs = req.timeoutMs ?? defaultTimeoutMs;
    const blobResponse = req.responseType === "blob";
    const headers = new Headers({ Accept: blobResponse ? "image/*" : "application/json" });
    const token = tokenSource();
    if (token) headers.set("Authorization", `Bearer ${token}`);

    let body: BodyInit | undefined;
    if (req.formData) {
      body = req.formData;
    } else if (req.json !== undefined) {
      body = JSON.stringify(req.json);
      headers.set("Content-Type", "application/json");
    }

    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    const onCallerAbort = () => controller.abort();
    req.signal?.addEventListener("abort", onCallerAbort, { once: true });

    let response: Response;
    try {
      response = await doFetch(`${baseUrl}${API_PREFIX}${path}`, {
        method: req.method ?? "GET",
        headers,
        body,
        signal: controller.signal,
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch (cause) {
      if (timedOut) {
        throw new ApiError("timeout", `Request timed out after ${timeoutMs} ms`, undefined, { cause });
      }
      throw new ApiError("network", "Could not reach the Nitid Studio server", undefined, { cause });
    } finally {
      clearTimeout(timer);
      req.signal?.removeEventListener("abort", onCallerAbort);
    }

    if (blobResponse && response.ok) {
      return (await response.blob()) as T;
    }

    let parsed: unknown;
    let parseFailed = false;
    const text = await response.text();
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parseFailed = true;
      }
    }

    if (!response.ok) {
      const kind: ApiErrorKind =
        response.status === 401 || response.status === 403 ? "unauthorized" : "http";
      const { code, message } = parseErrorBody(response.status, parsed);
      throw new ApiError(kind, message, response.status, code ? { code } : undefined);
    }
    if (parseFailed) {
      throw new ApiError("invalid-response", "Server returned invalid JSON", response.status);
    }
    return parsed as T;
  }

  return { request };
}
