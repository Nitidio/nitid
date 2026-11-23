/**
 * Light / dark theme of the Nitid app, the same switch as nitid.io: the system preference until the visitor chooses.
 * The choice lives in sessionStorage (per tab): the repo policy forbids persistent storage (docs/studio-development.md).
 */
export type Theme = "light" | "dark";

export const THEME_KEY = "nitid-theme";

/** The stored choice, or null (system preference). Storage may be unavailable or empty: never throws. */
export function readStoredTheme(storage: Pick<Storage, "getItem"> | null = safeStorage()): Theme | null {
  try {
    const value = storage?.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function storeTheme(theme: Theme, storage: Pick<Storage, "setItem"> | null = safeStorage()): void {
  try {
    storage?.setItem(THEME_KEY, theme);
  } catch {
    // A private window or a blocked store: the choice then lasts for this page only.
  }
}

/** The theme that is on screen now: the stored choice, else the system preference. */
export function effectiveTheme(
  stored: Theme | null,
  prefersDark: boolean = typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches,
): Theme {
  return stored ?? (prefersDark ? "dark" : "light");
}

export function applyTheme(theme: Theme | null, root: HTMLElement = document.documentElement): void {
  if (theme === null) root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}

function safeStorage(): Storage | null {
  try {
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null;
  }
}
