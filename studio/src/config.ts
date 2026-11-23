/**
 * Deployment configuration that is not discovered at run time.
 *
 * `__STUDIO_MODELS_BASE_URL__` is injected by `vite.config.ts` from the environment variable
 * `STUDIO_MODELS_BASE_URL` at build time. Default: empty, which means "no models configured":
 * the hosted (browser-only) site then shows an explanatory empty state. No model URL is
 * hardcoded anywhere in the source.
 *
 * The value is either root-relative (`/models/`, same origin as the app) or an absolute HTTPS URL
 * (the models live on another origin; docs/studio-hosted.md).
 */

/** Name of the catalogue file expected under the base URL. */
export const HOSTED_CATALOG_FILE = "models.json";

/** Normalised base URL (always ends with "/"), or `null` when none is configured. */
export function normalizeBaseUrl(value: string | null | undefined): string | null {
  const v = (value ?? "").trim();
  if (!v) return null;
  return v.endsWith("/") ? v : `${v}/`;
}

/** The configured hosted models base URL, or `null`. */
export function hostedModelsBaseUrl(): string | null {
  return normalizeBaseUrl(__STUDIO_MODELS_BASE_URL__);
}

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]"]);

/**
 * Validates `STUDIO_MODELS_BASE_URL` at build time and returns it unchanged ("" when not set).
 * Accepts: empty, a root-relative path (`/models/`), or an absolute `https:` URL. `http:` is accepted only
 * for loopback hosts (local development and the smoke test); an https page cannot load plain-http models.
 * Refuses credentials, a query string and a fragment: they would be dropped or leak into every request.
 */
export function validateModelsBaseUrl(value: string | null | undefined): string {
  const v = (value ?? "").trim();
  if (!v) return "";
  const fail = (why: string): never => {
    throw new Error(`STUDIO_MODELS_BASE_URL=${JSON.stringify(v)} is invalid: ${why}`);
  };
  if (v.startsWith("//")) return fail("a protocol-relative URL is ambiguous; use /path/ or an absolute https URL");
  if (v.startsWith("/")) {
    if (/[?#]/.test(v)) return fail("no query string or fragment is allowed");
    return v;
  }
  let url: URL;
  try {
    url = new URL(v);
  } catch {
    return fail("expected /path/ or an absolute https URL");
  }
  if (url.protocol !== "https:" && !(url.protocol === "http:" && LOOPBACK.has(url.hostname))) {
    return fail("an absolute URL must use https (http is only accepted for localhost)");
  }
  if (url.username || url.password) return fail("credentials in the URL are not allowed");
  if (url.search || url.hash) return fail("no query string or fragment is allowed");
  return v;
}
