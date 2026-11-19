/**
 * Deployment configuration that is not discovered at run time.
 *
 * `__STUDIO_MODELS_BASE_URL__` is injected by `vite.config.ts` from the environment variable
 * `STUDIO_MODELS_BASE_URL` at build time. Default: empty, which means "no models configured":
 * the hosted (browser-only) site then shows an explanatory empty state. No model URL is
 * hardcoded anywhere in the source.
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
