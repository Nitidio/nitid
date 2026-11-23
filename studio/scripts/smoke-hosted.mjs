/* global window, document */
// Browser smoke test of the HOSTED deployment, with no Python backend at all.
//
//   cd studio && pnpm run smoke:hosted              # both modes, positive and negative cases
//   SMOKE_MODE=cross-origin pnpm run smoke:hosted   # only the production topology (models on another origin)
//   SMOKE_MODE=same-origin  pnpm run smoke:hosted   # only the small-bundle topology (models in the site)
//
// Everything runs with the same pieces that production uses:
//   - the UI is built with `vite build --mode hosted` and STUDIO_MODELS_BASE_URL (an absolute URL of the
//     models server in cross-origin mode, /models/ in same-origin mode);
//   - a fixture model bundle (a tiny ONNX built by scripts/helpers/tiny-model.mjs, models.json, SHA256SUMS,
//     a licence file) is checked and the site is written by scripts/assemble_hosted.py (the real deploy gate),
//     which renders hosting/cloudflare/_headers.template into the production `_headers`;
//   - the site is served by a static server that applies exactly that `_headers` and the single-page-app
//     fallback of Cloudflare (127.0.0.1:PORT1);
//   - cross-origin mode: the bundle is served by ANOTHER origin (127.0.0.1:PORT2) that sends ONLY what the owner
//     configures on his server: Access-Control-Allow-Origin: <app origin>, Cross-Origin-Resource-Policy:
//     cross-origin and the right Content-Type. No cache headers, no Expose-Headers, no preflight handling needed.
//   - real Chrome is driven with playwright-core.
//
// Positive scenarios assert: cross-origin isolation with SharedArrayBuffer, zero console errors, zero CSP
// violations, no request to /api/*, a detection from the catalogue model, results on screen, JSON export, a
// reload served from the Cache API and (cross-origin) that the model requests went to the models origin, as CORS
// requests, with no cookie, no custom header and no preflight, and that nothing of the bundle is in the site.
// NEGATIVE scenarios must fail in a recognisable way, each reported with a clear message:
//   - the models server without Access-Control-Allow-Origin  -> the browser blocks the catalogue (CORS);
//   - the models origin missing from connect-src of the site's headers -> the CSP blocks it.
// Exits non-zero on any failed check. Needs Node >= 22.13, python3 (standard library only), playwright-core
// (PLAYWRIGHT_CORE_PATH or the usual locations) and Chrome (CHROME_PATH, default /usr/bin/google-chrome).
// Not part of `pnpm run check` (it needs a browser). Other knobs: SMOKE_HEADERS=<template> (try another headers
// template, to prove a weaker policy makes this test fail), SMOKE_KEEP=1 (keep the temporary folder).
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { headersFor, parseHeadersFile } from "./helpers/cf-headers.mjs";
import { ONNX, PNG, SHA } from "./helpers/tiny-model.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const studioDir = resolve(here, "..");
const repoDir = resolve(studioDir, "..");
const MODE = process.env.SMOKE_MODE ?? "all";
if (!["all", "same-origin", "cross-origin"].includes(MODE)) {
  console.error(`SMOKE_MODE must be all, same-origin or cross-origin (got ${MODE})`);
  process.exit(2);
}

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
let scope = "";
function check(name, ok, extra = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${scope}${name}${extra ? ` (${extra})` : ""}`);
  if (!ok) failed = true;
}

/* ---------------- building blocks ---------------- */
const work = mkdtempSync(join(tmpdir(), "studio-hosted-smoke-"));

function buildUi(name, modelsBaseUrl) {
  const out = join(work, name);
  console.log(`building the hosted UI (STUDIO_MODELS_BASE_URL=${modelsBaseUrl}) into ${out} ...`);
  execFileSync("pnpm", ["run", "build:hosted"], {
    cwd: studioDir,
    stdio: "ignore",
    env: { ...process.env, STUDIO_OUT_DIR: out, STUDIO_MODELS_BASE_URL: modelsBaseUrl },
  });
  return out;
}

function writeBundle() {
  const dir = join(work, "bundle");
  mkdirSync(dir);
  writeFileSync(join(dir, "tiny-detect.onnx"), ONNX);
  writeFileSync(
    join(dir, "models.json"),
    JSON.stringify({ models: [{ id: "tiny", task: "detect", url: "tiny-detect.onnx", sha256: SHA, label: "Tiny hosted" }] }),
  );
  writeFileSync(join(dir, "SHA256SUMS"), `${SHA}  tiny-detect.onnx\n`);
  writeFileSync(join(dir, "LICENSE-FIXTURE.txt"), "Test fixture. Not a real model.\n");
  return dir;
}

/** Runs scripts/assemble_hosted.py (the real deploy gate); returns the site folder. */
function assembleSite(name, ui, extraArgs) {
  const out = join(work, name);
  execFileSync(
    process.env.PYTHON ?? "python3",
    [
      join(repoDir, "scripts", "assemble_hosted.py"),
      "--ui", ui, "--out", out,
      // SMOKE_HEADERS: try another headers template (proves that a weaker policy makes this test fail).
      ...(process.env.SMOKE_HEADERS ? ["--headers-template", resolve(process.env.SMOKE_HEADERS)] : []),
      ...extraArgs,
    ],
    { stdio: "inherit" },
  );
  return out;
}

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

/** The site: static files + the production `_headers` + the SPA fallback of Cloudflare, on 127.0.0.1. */
async function startSite(siteDir) {
  const rules = parseHeadersFile(readFileSync(join(siteDir, "_headers"), "utf8"));
  const requests = [];
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const entry = { method: req.method, path: url.pathname, cookie: req.headers.cookie };
    requests.push(entry);
    let file = resolve(siteDir, `.${normalize(url.pathname)}`);
    let info = null;
    if (file.startsWith(siteDir + "/") && !file.endsWith("/_headers")) info = await stat(file).catch(() => null);
    if (!info?.isFile()) {
      // `not_found_handling: "single-page-application"`: any unknown path answers index.html with 200.
      file = join(siteDir, "index.html");
      entry.fallback = true;
    }
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
      "Content-Length": body.length,
      ...headersFor(rules, url.pathname),
    });
    res.end(req.method === "HEAD" ? undefined : body);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  return { origin: `http://127.0.0.1:${server.address().port}`, requests, close: () => server.close() };
}

/**
 * The models server of the owner. It sends ONLY what docs/studio-hosted-models.md tells him to configure:
 * Access-Control-Allow-Origin (the app's origin), Cross-Origin-Resource-Policy: cross-origin and the right
 * Content-Type; GET, HEAD and OPTIONS are answered. `state` switches the headers off for the negative cases.
 */
async function startModelsServer(bundleDir) {
  const state = { appOrigin: null, acao: true, corp: true, log: [] };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    state.log.push({ method: req.method, path: url.pathname, headers: req.headers });
    const headers = {};
    if (state.acao && state.appOrigin) headers["Access-Control-Allow-Origin"] = state.appOrigin;
    if (state.corp) headers["Cross-Origin-Resource-Policy"] = "cross-origin";
    if (req.method === "OPTIONS") {
      res.writeHead(204, { ...headers, "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS", "Access-Control-Max-Age": "600" });
      return res.end();
    }
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, { ...headers, Allow: "GET, HEAD, OPTIONS" });
      return res.end();
    }
    const file = resolve(bundleDir, `.${normalize(url.pathname)}`);
    const info = file.startsWith(bundleDir + "/") ? await stat(file).catch(() => null) : null;
    if (!info?.isFile()) {
      res.writeHead(404, headers);
      return res.end();
    }
    const body = await readFile(file);
    res.writeHead(200, { ...headers, "Content-Type": MIME[extname(file)] ?? "application/octet-stream", "Content-Length": body.length });
    res.end(req.method === "HEAD" ? undefined : body);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const origin = `http://127.0.0.1:${server.address().port}`;
  return { origin, state, close: () => server.close() };
}

/** Listeners that record everything a page does. */
async function instrument(context, page) {
  const seen = { errors: [], failedLoads: [], responses: [], onnxGets: new Map() };
  await page.addInitScript(() => {
    window.__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
  });
  page.on("pageerror", (e) => seen.errors.push(`pageerror: ${e}`));
  page.on("console", (m) => m.type() === "error" && seen.errors.push(`console: ${m.text()}`));
  page.on("requestfailed", (r) => seen.failedLoads.push(`failed ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => {
    const u = new URL(r.url());
    seen.responses.push({ path: u.pathname, origin: u.origin, status: r.status(), headers: r.headers() });
    if (r.status() >= 400) seen.failedLoads.push(`${r.status()} ${u.origin}${u.pathname}`);
    if (u.pathname.endsWith(".onnx") && r.request().method() === "GET") seen.onnxGets.set(u.origin, (seen.onnxGets.get(u.origin) ?? 0) + 1);
  });
  return seen;
}

/** What a failure most likely means, from what Chrome said. For the owner reading a failed run. */
function diagnose(errors, csp) {
  const lines = [];
  if (errors.some((e) => /blocked by CORS policy/.test(e) && /Access-Control-Allow-Origin/.test(e))) {
    lines.push("the models server does not send Access-Control-Allow-Origin for this app's origin (CORS)");
  }
  if (csp.some((v) => v.startsWith("connect-src"))) {
    lines.push(`the Content-Security-Policy blocked a request: ${csp.filter((v) => v.startsWith("connect-src")).join(", ")} (the models origin is missing from connect-src)`);
  }
  if (errors.some((e) => /ERR_BLOCKED_BY_RESPONSE|NotSameOriginAfterDefaultedToSameOriginByCoep|Cross-Origin-Resource-Policy/.test(e))) {
    lines.push("a response was blocked by COEP/CORP: the resource needs CORS or Cross-Origin-Resource-Policy: cross-origin");
  }
  return lines.length ? `DIAGNOSIS: ${lines.join("; ")}` : "";
}

/* ---------------- the positive scenario ---------------- */
/**
 * @param {object} p
 * @param {import("playwright-core").Browser} p.browser
 * @param {{origin: string, requests: object[]}} p.site
 * @param {{origin: string, state: object} | null} p.models  null = the models are in the site (same-origin mode)
 */
async function positiveScenario({ browser, site, models }) {
  const origin = site.origin;
  const modelsOrigin = models ? models.origin : origin;
  const catalogPath = models ? "/models.json" : "/models/models.json";
  const onnxPath = models ? "/tiny-detect.onnx" : "/models/tiny-detect.onnx";
  const cross = models !== null;
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const seen = await instrument(context, page);
  // A cookie for the host (cookies ignore the port): it would be sent to the models server if the app
  // fetched the models with credentials. The site's own server is the control: it must see it.
  await context.addCookies([{ name: "probe", value: "1", url: origin }]);
  try {
    const home = await page.goto(`${origin}/`);
    const homeHeaders = home.headers();
    check("served with COOP same-origin and COEP require-corp", homeHeaders["cross-origin-opener-policy"] === "same-origin" && homeHeaders["cross-origin-embedder-policy"] === "require-corp");
    check("the app's own files keep CORP same-origin", homeHeaders["cross-origin-resource-policy"] === "same-origin", homeHeaders["cross-origin-resource-policy"]);
    check("index.html is not cached", homeHeaders["cache-control"] === "no-cache", homeHeaders["cache-control"]);
    const csp = homeHeaders["content-security-policy"] ?? "";
    const connectSrc = /connect-src ([^;]*)/.exec(csp)?.[1];
    check(
      cross ? "CSP connect-src is exactly 'self' plus the models origin" : "CSP connect-src is 'self' only",
      connectSrc === (cross ? `'self' ${modelsOrigin}` : "'self'"),
      connectSrc,
    );

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
    check("a client route is answered with index.html (SPA fallback)", site.requests.some((r) => r.path === "/run/browser" && r.fallback));
    await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
    await page.getByRole("button", { name: "Run" }).click();
    await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
    const rows = await page.locator("table.detections tbody tr").count();
    const gets = seen.onnxGets.get(modelsOrigin) ?? 0;
    check("a detection ran in the browser from the catalogue model: 2 detections", rows === 2 && gets === 1, `rows=${rows} onnxGets=${gets}`);
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
    const gets2 = seen.onnxGets.get(modelsOrigin) ?? 0;
    check("after a reload the model comes from the Cache API (no second download)", gets2 === 1, `onnxGets=${gets2}`);

    /* ---- network and policy assertions over everything that happened ---- */
    const api = site.requests.filter((r) => r.path.startsWith("/api/"));
    check("no request to any /api/* path", api.length === 0, api.map((r) => r.path).join(","));
    const allowed = new Set(cross ? [origin, modelsOrigin] : [origin]);
    const foreign = seen.responses.filter((r) => !allowed.has(r.origin) && !r.path.startsWith("blob:"));
    check(cross ? "every response came from the site or from the models origin, nothing else" : "every response came from the site's own origin", foreign.length === 0, foreign.map((r) => r.origin + r.path).join(","));
    const wasm = seen.responses.find((r) => r.path.endsWith(".wasm"));
    check("the ORT wasm was loaded from /ort/ of the site as application/wasm", Boolean(wasm) && wasm.origin === origin && wasm.path.startsWith("/ort/") && wasm.headers["content-type"] === "application/wasm", wasm ? `${wasm.path} ${wasm.headers["content-type"]}` : "no wasm request");
    const asset = seen.responses.find((r) => r.path.startsWith("/assets/") && r.path.endsWith(".js"));
    check("hashed assets are immutable", asset?.headers["cache-control"] === "public, max-age=31536000, immutable", asset?.headers["cache-control"]);
    const catalog = seen.responses.find((r) => r.path === catalogPath && r.origin === modelsOrigin);
    if (cross) {
      check("models.json was fetched from the models origin, not from the site", catalog?.status === 200 && !site.requests.some((r) => r.path.endsWith("models.json")), `${modelsOrigin}${catalogPath} ${catalog?.status}`);
    } else {
      check("models.json is served with no-cache", catalog?.status === 200 && catalog.headers["cache-control"] === "no-cache", catalog?.headers["cache-control"]);
    }
    const missingHeader = seen.responses.filter((r) => r.origin === origin && r.headers["cross-origin-embedder-policy"] !== "require-corp");
    check("every response of the site carries COEP", missingHeader.length === 0, missingHeader.map((r) => r.path).join(","));

    if (cross) {
      const log = models.state.log;
      const modelRequests = log.filter((r) => r.path === catalogPath || r.path === onnxPath);
      check("the model requests went to the models origin: models.json and the ONNX (GET)", modelRequests.some((r) => r.path === catalogPath && r.method === "GET") && modelRequests.some((r) => r.path === onnxPath && r.method === "GET"), modelRequests.map((r) => `${r.method} ${r.path}`).join(", "));
      check("nothing of the bundle is in the site: no .onnx or models.json requested from it", !site.requests.some((r) => r.path.endsWith(".onnx") || r.path.endsWith("models.json")));
      check("the models server saw only GET (no preflight, no HEAD: the sha256 is in the catalogue)", log.every((r) => r.method === "GET"), [...new Set(log.map((r) => r.method))].join(","));
      check("the requests are CORS requests from the app's origin", modelRequests.every((r) => r.headers.origin === origin && r.headers["sec-fetch-mode"] === "cors"), modelRequests.map((r) => `${r.headers.origin} ${r.headers["sec-fetch-mode"]}`).join(" | "));
      check("the site's server saw the probe cookie (the control: cookies exist for this host)", site.requests.some((r) => r.cookie?.includes("probe=1")));
      check("no credentials: the models server never received a cookie or an Authorization header", log.every((r) => !r.headers.cookie && !r.headers.authorization), log.map((r) => Object.keys(r.headers).filter((h) => h === "cookie" || h === "authorization").join("")).join(""));
      check("the models server's catalogue and ONNX carry the CORS and CORP headers it was configured with", seen.responses.filter((r) => r.origin === modelsOrigin && (r.path === catalogPath || r.path === onnxPath)).every((r) => r.status === 200 && r.headers["access-control-allow-origin"] === origin && (!models.state.corp || r.headers["cross-origin-resource-policy"] === "cross-origin")));
    }
    const violations = await page.evaluate(() => window.__csp);
    check("no Content-Security-Policy violation", violations.length === 0, violations.join(" | "));
    check("no console or page errors (favicon included)", seen.errors.length === 0, seen.errors.join(" | "));
    check("no failed requests", seen.failedLoads.length === 0, seen.failedLoads.join(" | "));
    if (seen.errors.length) console.log(diagnose(seen.errors, violations));
  } catch (error) {
    // A policy or runtime failure usually shows up as a timeout: report it as a failed check, not a crash.
    check("the scenario ran to the end", false, String(error.message ?? error).split("\n")[0]);
    const violations = await page.evaluate(() => window.__csp).catch(() => []);
    if (seen.errors.length || violations.length) {
      console.log(`console errors so far: ${seen.errors.join(" | ")}`);
      console.log(diagnose(seen.errors, violations));
    }
  } finally {
    await context.close();
  }
}

/* ---------------- the negative scenarios ---------------- */
/**
 * The app must NOT load the catalogue, must say so, and Chrome must name the reason.
 * @param {"cors"|"csp"} reason
 */
async function negativeScenario({ browser, site, models, reason }) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const seen = await instrument(context, page);
  try {
    await page.goto(`${site.origin}/run/browser`);
    await page.getByText("Could not get the model list.").waitFor({ timeout: 20000 });
    const optionLoaded = await page.getByRole("option", { name: "Tiny hosted" }).count();
    check("the model list is NOT loaded", optionLoaded === 0);
    const notice = (await page.locator("body").innerText()).split("\n").find((l) => /catalogue/i.test(l)) ?? "";
    check("the app tells the visitor that the catalogue could not be reached", /could not be reached/.test(notice), notice);
    const violations = await page.evaluate(() => window.__csp);
    const hint = diagnose(seen.errors, violations);
    if (reason === "cors") {
      const message = seen.errors.find((e) => /blocked by CORS policy/.test(e)) ?? "";
      check("Chrome blocked it for CORS: 'No Access-Control-Allow-Origin header'", /No 'Access-Control-Allow-Origin' header/.test(message), message.slice(0, 220));
      check("the request did reach the models server (it is the response that was refused)", models.state.log.some((r) => r.path === "/models.json"));
      check("clear diagnosis for the owner", hint.includes("does not send Access-Control-Allow-Origin"), hint);
    } else {
      const csp = violations.find((v) => v.startsWith("connect-src")) ?? "";
      check("Chrome blocked it by the CSP: connect-src violation for the models URL", csp.includes(`${models.origin}/models.json`), csp);
      check("the request never left the browser (the models server saw nothing)", models.state.log.length === 0, `${models.state.log.length} requests`);
      check("clear diagnosis for the owner", hint.includes("missing from connect-src"), hint);
    }
    console.log(`     -> ${hint}`);
  } catch (error) {
    check("the app reported the failure instead of loading the models", false, String(error.message ?? error).split("\n")[0]);
  } finally {
    await context.close();
  }
}

/* ---------------- run ---------------- */
const { chromium } = loadPlaywright();
const bundleDir = writeBundle();
const servers = [];
let browser;
try {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome", args: ["--no-sandbox"] });

  if (MODE === "all" || MODE === "same-origin") {
    scope = "[same-origin] ";
    console.log("\n== same-origin: the bundle is copied into the site (small models only) ==");
    const ui = buildUi("ui-same", "/models/");
    const siteDir = assembleSite("site-same", ui, ["--bundle", bundleDir]);
    const site = await startSite(siteDir);
    servers.push(site);
    await positiveScenario({ browser, site, models: null });
  }

  if (MODE === "all" || MODE === "cross-origin") {
    console.log("\n== cross-origin: the site on one origin, the bundle on another ==");
    // The models server comes first: its address is baked into the UI build and into the CSP.
    const models = await startModelsServer(bundleDir);
    servers.push(models);
    const base = `${models.origin}/`;
    const ui = buildUi("ui-cross", base);
    const siteDir = assembleSite("site-cross", ui, [
      "--models-base-url", base, "--allow-http-loopback", "--validate-bundle", bundleDir, "--bundle-check", "basic",
    ]);
    const site = await startSite(siteDir);
    servers.push(site);
    models.state.appOrigin = site.origin; // what the owner configures: the app's origin

    scope = "[cross-origin] ";
    await positiveScenario({ browser, site, models });

    scope = "[cross-origin, ACAO only] ";
    console.log("\n-- variant: the models server sends Access-Control-Allow-Origin but NOT Cross-Origin-Resource-Policy --");
    console.log("   (the app fetches in CORS mode, which COEP accepts without CORP; the docs still ask for CORP)");
    models.state.corp = false;
    models.state.log.length = 0;
    await positiveScenario({ browser, site, models });
    models.state.corp = true;

    scope = "[NEGATIVE: no Access-Control-Allow-Origin] ";
    console.log("\n-- negative: the models server does not send Access-Control-Allow-Origin (must be refused) --");
    models.state.acao = false;
    models.state.log.length = 0;
    await negativeScenario({ browser, site, models, reason: "cors" });
    models.state.acao = true;

    scope = "[NEGATIVE: origin missing from connect-src] ";
    console.log("\n-- negative: the site's headers do not list the models origin in connect-src (must be refused) --");
    const brokenDir = join(work, "site-no-csp");
    cpSync(siteDir, brokenDir, { recursive: true });
    const headersPath = join(brokenDir, "_headers");
    const original = readFileSync(headersPath, "utf8");
    const patched = original.replace(`connect-src 'self' ${models.origin}`, "connect-src 'self'");
    check("test setup: the broken headers differ from the generated ones only by the models origin", patched !== original && patched.replace("connect-src 'self'", `connect-src 'self' ${models.origin}`) === original);
    writeFileSync(headersPath, patched);
    const brokenSite = await startSite(brokenDir);
    servers.push(brokenSite);
    models.state.appOrigin = brokenSite.origin;
    models.state.log.length = 0;
    await negativeScenario({ browser, site: brokenSite, models, reason: "csp" });
  }
} catch (error) {
  scope = "";
  check("the smoke test ran to the end", false, String(error.message ?? error).split("\n")[0]);
} finally {
  await browser?.close();
  for (const s of servers) s.close();
  if (!process.env.SMOKE_KEEP) rmSync(work, { recursive: true, force: true });
}
console.log(failed ? "smoke:hosted FAILED" : "smoke:hosted OK");
process.exit(failed ? 1 : 0);
