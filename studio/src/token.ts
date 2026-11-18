/**
 * Session token handling.
 *
 * The local server opens the browser at `127.0.0.1:PORT/#t=<token>`.
 * The fragment never reaches the server or any proxy; we read it once, remove
 * it from the address bar and keep the token only in memory, mirrored in
 * sessionStorage so a page reload inside the same tab keeps working.
 * Persistent browser storage is never used (the token must not outlive the tab).
 */

const STORAGE_KEY = "nitid-studio.token";
let memoryToken: string | null = null;

function readSession(): string | null {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // storage disabled
  }
}

function writeSession(token: string | null): void {
  try {
    if (token === null) window.sessionStorage.removeItem(STORAGE_KEY);
    else window.sessionStorage.setItem(STORAGE_KEY, token);
  } catch {
    /* storage disabled: memory only */
  }
}

/** Extract `t` from a URL fragment such as `#t=abc`. Returns null if absent or empty. */
export function parseTokenFragment(hash: string): string | null {
  const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
  const token = params.get("t");
  return token ? token : null;
}

/**
 * Call once at start-up, before the router mounts. If the URL carries a token
 * fragment, store it and strip the fragment with `history.replaceState`
 * (no navigation, no new history entry).
 */
export function initToken(loc: Location = window.location, hist: History = window.history): void {
  const fromUrl = parseTokenFragment(loc.hash);
  if (fromUrl !== null) {
    memoryToken = fromUrl;
    writeSession(fromUrl);
    hist.replaceState(hist.state, "", loc.pathname + loc.search);
  } else if (memoryToken === null) {
    memoryToken = readSession();
  }
}

export function getToken(): string | null {
  return memoryToken;
}

export function clearToken(): void {
  memoryToken = null;
  writeSession(null);
}
