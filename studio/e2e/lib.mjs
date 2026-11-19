// Shared helpers of the Studio end-to-end drivers (Node + playwright-core + real Chrome).
// The Python tests start the real `nitid studio` process and call these drivers with its URL;
// each driver writes a JSON result file {checks: [...], data: {...}} that Python asserts on.
import { createRequire } from "node:module";
import { existsSync, writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);

/** Exit code that means "prerequisite missing, skip" (the Python wrapper maps it to a skip). */
export const EXIT_SKIP = 77;

export function skip(reason) {
  console.log(`SKIP: ${reason}`);
  process.exit(EXIT_SKIP);
}

export function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_CORE_PATH,
    "playwright-core",
    "/opt/npm-tools/node_modules/playwright-core",
    "/opt/node-tools/node_modules/playwright-core",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch {
      /* try the next one */
    }
  }
  return skip("playwright-core not found (set PLAYWRIGHT_CORE_PATH)");
}

export function chromePath() {
  const candidates = [
    process.env.CHROME_PATH,
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  const found = candidates.find((path) => existsSync(path));
  return found ?? skip("no Chrome/Chromium found (set CHROME_PATH)");
}

/** Launch headless Chrome. `args` are extra command-line switches. */
export async function launchChrome(args = []) {
  const { chromium } = loadPlaywright();
  return chromium.launch({
    executablePath: chromePath(),
    args: ["--no-sandbox", ...args],
  });
}

export function parseArgs(argv = process.argv.slice(2)) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key?.startsWith("--") || argv[i + 1] === undefined) throw new Error(`bad arguments near ${key}`);
    out[key.slice(2)] = argv[i + 1];
  }
  return out;
}

/** Split `http://127.0.0.1:PORT/#t=TOKEN` into its parts. */
export function splitStudioUrl(url) {
  const parsed = new URL(url);
  const token = new URLSearchParams(parsed.hash.slice(1)).get("t");
  return { origin: parsed.origin, port: parsed.port, token, url };
}

/** Collects named checks and free-form data, and writes them as JSON. */
export class Recorder {
  constructor(outPath) {
    this.outPath = outPath;
    this.checks = [];
    this.data = {};
  }

  check(name, ok, detail = "") {
    const passed = Boolean(ok);
    this.checks.push({ name, ok: passed, detail: String(detail) });
    console.log(`${passed ? "PASS" : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
    return passed;
  }

  finish() {
    writeFileSync(this.outPath, JSON.stringify({ checks: this.checks, data: this.data }, null, 2));
    process.exit(this.checks.every((c) => c.ok) ? 0 : 1);
  }
}

/**
 * Track console errors, page errors and failed requests of a page.
 * `expected` lists substrings of messages that are part of the scenario (e.g. a 401 on purpose).
 */
export function watchPage(page) {
  const problems = [];
  page.on("pageerror", (error) => problems.push(`pageerror: ${error}`));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(`console.error: ${message.text()}`);
  });
  page.on("response", (response) => {
    if (response.status() >= 400) problems.push(`http ${response.status()}: ${response.url()}`);
  });
  page.on("requestfailed", (request) =>
    problems.push(`requestfailed: ${request.url()} ${request.failure()?.errorText ?? ""}`),
  );
  return {
    problems,
    unexpected(expected = []) {
      return problems.filter((p) => !expected.some((e) => p.includes(e)));
    },
  };
}
