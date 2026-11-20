/* global window, document */
// Browser smoke test of the HOSTED deployment, with no Python backend at all.
//
//   cd studio && pnpm run smoke:hosted
//
// What it does, end to end, with the same pieces that production uses:
//   1. builds the UI with `vite build --mode hosted` and STUDIO_MODELS_BASE_URL=/models/ (or reuses
//      SMOKE_HOSTED_UI);
//   2. writes a fixture model bundle (a tiny ONNX built by scripts/lib/tiny-model.mjs, models.json,
//      SHA256SUMS, a licence file);
//   3. runs scripts/assemble_hosted.py on both (the real deploy gate) into a temporary site folder;
//   4. serves that folder with a static server that applies hosting/cloudflare/_headers (the production
//      file, read from the assembled folder) and the single-page-app fallback of Cloudflare;
//   5. drives real Chrome and asserts: cross-origin isolation with SharedArrayBuffer, no CSP violation,
//      zero console errors, no request to /api/*, no request to another origin, application/wasm served,
//      a detection from the catalogue model, results on screen, JSON export, a reload served from the cache.
// Exits non-zero on any failed check. Needs Node >= 22.13, python3 (standard library only), playwright-core
// (PLAYWRIGHT_CORE_PATH or the usual locations) and Chrome (CHROME_PATH, default /usr/bin/google-chrome).
// Not part of `pnpm run check` (it needs a browser).
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { headersFor, parseHeadersFile } from "./lib/cf-headers.mjs";
import { ONNX, PNG, SHA } from "./lib/tiny-model.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const studioDir = resolve(here, "..");
const repoDir = resolve(studioDir, "..");

const require = createRequire(import.meta.url);
function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_CORE_PATH,
    "playwright-core",
    "/opt/npm-tools/node_modules/playwright-core",
    "/opt/node-tools/node_modules/playwright-core",
  ].filter(Boolean);
  for (const c of candidates) {
    try {
      return require(c);
    } catch {
      /* try next */
    }
  }
  throw new Error("playwright-core not found; set PLAYWRIGHT_CORE_PATH");
}

let failed = false;
function check(name, ok, extra = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${extra ? ` (${extra})` : ""}`);
  if (!ok) failed = true;
}

/* ---------------- 1-3. build, bundle, assemble ---------------- */
const work = mkdtempSync(join(tmpdir(), "studio-hosted-smoke-"));
const uiDir = process.env.SMOKE_HOSTED_UI ? resolve(process.env.SMOKE_HOSTED_UI) : join(work, "ui");
if (!process.env.SMOKE_HOSTED_UI) {
  console.log(`building the hosted UI into ${uiDir} ...`);
  execFileSync("pnpm", ["run", "build:hosted"], {
    cwd: studioDir,
    stdio: "ignore",
    env: { ...process.env, STUDIO_OUT_DIR: uiDir, STUDIO_MODELS_BASE_URL: "/models/" },
  });
}

const bundleDir = join(work, "bundle");
mkdirSync(bundleDir);
writeFileSync(join(bundleDir, "tiny-detect.onnx"), ONNX);
writeFileSync(
  join(bundleDir, "models.json"),
  JSON.stringify({ models: [{ id: "tiny", task: "detect", url: "tiny-detect.onnx", sha256: SHA, label: "Tiny hosted" }] }),
);
writeFileSync(join(bundleDir, "SHA256SUMS"), `${SHA}  tiny-detect.onnx\n`);
writeFileSync(join(bundleDir, "LICENSE-FIXTURE.txt"), "Test fixture. Not a real model.\n");

const siteDir = join(work, "site");
execFileSync(
  process.env.PYTHON ?? "python3",
  [
    join(repoDir, "scripts", "assemble_hosted.py"),
    "--ui", uiDir, "--bundle", bundleDir, "--out", siteDir,
    // SMOKE_HEADERS: try another _headers file (used to prove that a weaker policy makes this test fail).
    ...(process.env.SMOKE_HEADERS ? ["--headers", resolve(process.env.SMOKE_HEADERS)] : []),
  ],
  { stdio: "inherit" },
);

/* ---------------- 4. a static server with the production headers ---------------- */
const rules = parseHeadersFile(readFileSync(join(siteDir, "_headers"), "utf8"));
// Content types as Cloudflare infers them from the extension (checked here, not assumed).
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".txt": "text/plain; charset=utf-8",
};
const requests = [];
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  requests.push({ method: req.method, path: url.pathname });
  let file = resolve(siteDir, `.${normalize(url.pathname)}`);
  let fallback = false;
  let info = null;
  if (file.startsWith(siteDir + "/") && !file.endsWith("/_headers")) info = await stat(file).catch(() => null);
  if (!info?.isFile()) {
    // `not_found_handling: "single-page-application"`: any unknown path answers index.html with 200.
    file = join(siteDir, "index.html");
    fallback = true;
  }
  const body = await readFile(file);
  res.writeHead(200, {
    "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
    "Content-Length": body.length,
    ...headersFor(rules, url.pathname),
  });
  res.end(req.method === "HEAD" ? undefined : body);
  if (fallback) requests[requests.length - 1].fallback = true;
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

/* ---------------- 5. the browser ---------------- */
const { chromium } = loadPlaywright();
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome", args: ["--no-sandbox"] });
const errors = [];
const failedLoads = [];
const responses = [];
let onnxGets = 0;

try {
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  await page.addInitScript(() => {
    window.__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
  page.on("requestfailed", (r) => failedLoads.push(`failed ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => {
    const u = new URL(r.url());
    responses.push({ path: u.pathname, origin: u.origin, status: r.status(), headers: r.headers() });
    if (r.status() >= 400) failedLoads.push(`${r.status()} ${u.pathname}`);
    if (u.pathname === "/models/tiny-detect.onnx" && r.request().method() === "GET") onnxGets += 1;
  });

  const home = await page.goto(`${origin}/`);
  const homeHeaders = home.headers();
  check("served with COOP same-origin and COEP require-corp", homeHeaders["cross-origin-opener-policy"] === "same-origin" && homeHeaders["cross-origin-embedder-policy"] === "require-corp");
  check("index.html is not cached", homeHeaders["cache-control"] === "no-cache", homeHeaders["cache-control"]);

  await page.getByRole("button", { name: "Use browser mode" }).waitFor();
  const isolation = await page.evaluate(() => ({
    isolated: window.crossOriginIsolated,
    sab: typeof SharedArrayBuffer === "function",
    cores: navigator.hardwareConcurrency,
  }));
  check("page is crossOriginIsolated", isolation.isolated === true);
  check("SharedArrayBuffer is available (multi-threaded WebAssembly)", isolation.sab === true, `${isolation.cores} cores`);

  const nav = await page.getByRole("navigation", { name: "Main" }).getByRole("link").allTextContents();
  check("hosted navigation: no Local, no Compare", nav.join(",") === "Home,Browser,About", nav.join(","));
  check("the local-mode card is disabled", await page.getByRole("button", { name: "Use local mode" }).isDisabled());

  // A deep link is answered by the single-page-app fallback (like Cloudflare), then the router takes over.
  await page.goto(`${origin}/run/browser`);
  await page.getByRole("option", { name: "Tiny hosted" }).waitFor({ state: "attached", timeout: 15000 });
  check("a client route is answered with index.html (SPA fallback)", requests.some((r) => r.path === "/run/browser" && r.fallback));
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
  const rows = await page.locator("table.detections tbody tr").count();
  check("a detection ran in the browser from the catalogue model: 2 detections", rows === 2 && onnxGets === 1, `rows=${rows} onnxGets=${onnxGets}`);
  const canvas = page.getByRole("img", { name: /white\.png: 2 detections shown/ });
  await canvas.waitFor();
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas.detection-canvas");
    const d = c.getContext("2d").getImageData(90, 50, 1, 1).data;
    return d[0] === 255 && d[1] === 255 && d[2] === 255;
  });
  const edge = await canvas.evaluate((c) => Array.from(c.getContext("2d").getImageData(25, 40, 2, 1).data));
  check("boxes are drawn on the image", edge.slice(0, 3).some((v) => v < 200) || edge.slice(4, 7).some((v) => v < 200), JSON.stringify(edge));

  await page.getByRole("button", { name: "Export" }).click();
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download JSON" }).click()]);
  const doc = JSON.parse(readFileSync(await download.path(), "utf8"));
  check("JSON export works and records the browser engine", doc.source === "browser" && doc.detection_count === 2 && doc.runtime.backend === "wasm", download.suggestedFilename());

  await page.reload();
  await page.getByRole("option", { name: "Tiny hosted" }).waitFor({ state: "attached", timeout: 15000 });
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
  check("after a reload the model comes from the Cache API (no second download)", onnxGets === 1, `onnxGets=${onnxGets}`);

  /* ---- network and policy assertions over everything that happened ---- */
  const api = requests.filter((r) => r.path.startsWith("/api/"));
  check("no request to any /api/* path", api.length === 0, api.map((r) => r.path).join(","));
  const foreign = responses.filter((r) => r.origin !== origin && !r.path.startsWith("blob:"));
  check("every response came from the site's own origin", foreign.length === 0, foreign.map((r) => r.origin + r.path).join(","));
  const wasm = responses.find((r) => r.path.endsWith(".wasm"));
  check("the ORT wasm was loaded from /ort/ as application/wasm", Boolean(wasm) && wasm.path.startsWith("/ort/") && wasm.headers["content-type"] === "application/wasm", wasm ? `${wasm.path} ${wasm.headers["content-type"]}` : "no wasm request");
  const asset = responses.find((r) => r.path.startsWith("/assets/") && r.path.endsWith(".js"));
  check("hashed assets are immutable", asset?.headers["cache-control"] === "public, max-age=31536000, immutable", asset?.headers["cache-control"]);
  const catalog = responses.find((r) => r.path === "/models/models.json");
  check("models.json is served with no-cache", catalog?.status === 200 && catalog.headers["cache-control"] === "no-cache", catalog?.headers["cache-control"]);
  const missingHeader = responses.filter((r) => r.origin === origin && r.headers["cross-origin-embedder-policy"] !== "require-corp");
  check("every response carries COEP", missingHeader.length === 0, missingHeader.map((r) => r.path).join(","));
  const violations = await page.evaluate(() => window.__csp);
  check("no Content-Security-Policy violation", violations.length === 0, violations.join(" | "));
  check("no console or page errors (favicon included)", errors.length === 0, errors.join(" | "));
  check("no failed requests", failedLoads.length === 0, failedLoads.join(" | "));
} catch (error) {
  // A policy or runtime failure usually shows up as a timeout: report it as a failed check, not a crash.
  check("the scenario ran to the end", false, String(error.message ?? error).split("\n")[0]);
  if (errors.length) console.log(`console errors so far: ${errors.join(" | ")}`);
} finally {
  await browser.close();
  server.close();
  if (!process.env.SMOKE_KEEP) rmSync(work, { recursive: true, force: true });
}
console.log(failed ? "smoke:hosted FAILED" : "smoke:hosted OK");
process.exit(failed ? 1 : 0);
