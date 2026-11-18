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

  constructor(kind: ApiErrorKind, message: string, status?: number, options?: ErrorOptions) {
    super(message, options);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
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
}

export interface ApiClient {
  request<T>(path: string, options?: RequestOptions): Promise<T>;
}

function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const { detail, message } = body as ApiErrorBody;
    if (typeof detail === "string" && detail) return detail;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed with status ${status}`;
}

/** Same-origin JSON client that adds the bearer token and maps failures to `ApiError`. */
export function createApiClient(options: ClientOptions = {}): ApiClient {
  const baseUrl = options.baseUrl ?? "";
  const tokenSource = options.getToken ?? getToken;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  async function request<T>(path: string, req: RequestOptions = {}): Promise<T> {
    const doFetch = options.fetchImpl ?? fetch;
    const headers = new Headers({ Accept: "application/json" });
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
      throw new ApiError(kind, errorMessage(response.status, parsed), response.status);
    }
    if (parseFailed) {
      throw new ApiError("invalid-response", "Server returned invalid JSON", response.status);
    }
    return parsed as T;
  }

  return { request };
}
