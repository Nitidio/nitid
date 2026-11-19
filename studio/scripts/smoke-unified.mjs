/* global document */
// Browser smoke test of the unified UI against a mocked API. No Python backend, no network.
//
//   pnpm run build && pnpm run smoke:unified
//
// What it drives (headless Chrome, served with the target CSP
// `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:`):
//   1. LOCAL deployment (mock /api/v1): navigation, server-mode run, Export menu and a real JSON download.
//   2. In-browser mode with a REAL ONNX Runtime Web session: a tiny ONNX file that is built here
//      (constant outputs), served with ETag = sha256 and the bearer token. Checks the detections, the
//      canvas, the JSON export and that a reload takes the model from the Cache API (no second download).
//   3. Comparison view: server side vs browser side, agreement summary, keyboard activation.
//   4. HOSTED deployment (a second build with STUDIO_MODELS_BASE_URL, no /api): navigation without
//      server-only entries, catalogue -> model -> run; and the empty state of the default build.
// Exits non-zero on any failed check or any console/page error. Needs playwright-core
// (PLAYWRIGHT_CORE_PATH or the usual global locations) and Chrome (CHROME_PATH, default /usr/bin/google-chrome).
// Not part of `pnpm run check` (no new dependency).
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, normalize, resolve } from "node:path";
import { deflateSync } from "node:zlib";

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
const { chromium } = loadPlaywright();

const localDist = resolve(process.env.STUDIO_OUT_DIR ?? "../nitid/studio/static");
if (!existsSync(join(localDist, "index.html"))) throw new Error(`Build first: ${localDist}/index.html missing`);

/* ---------------- a 200x100 white PNG ---------------- */
function crc32(buf) {
  let c, crc = ~0;
  for (const b of buf) {
    c = (crc ^ b) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return ~crc >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function whitePng(w, h) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const row = Buffer.concat([Buffer.from([0]), Buffer.alloc(w * 3, 255)]);
  const raw = Buffer.concat(Array.from({ length: h }, () => row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}
const PNG = whitePng(200, 100);

/* ---------------- a tiny ONNX file, written as protobuf by hand ----------------
 * Constant outputs in the post-processed layout (labels, boxes, scores) for a 64x64 input, so the
 * result is known exactly and a real ONNX Runtime Web session runs it. No `onnx` package needed. */
const S = 64;
const varint = (value) => {
  let n = BigInt(value);
  const out = [];
  do {
    let b = Number(n & 0x7fn);
    n >>= 7n;
    if (n) b |= 0x80;
    out.push(b);
  } while (n);
  return Buffer.from(out);
};
const tag = (field, wire) => varint((field << 3) | wire);
const lenField = (field, buf) => Buffer.concat([tag(field, 2), varint(buf.length), buf]);
const strField = (field, s) => lenField(field, Buffer.from(s, "utf8"));
const intField = (field, n) => Buffer.concat([tag(field, 0), varint(n)]);
const tensor = (name, dims, dataType, raw) =>
  Buffer.concat([...dims.map((d) => intField(1, d)), intField(2, dataType), strField(8, name), lenField(9, raw)]);
const valueInfo = (name, elemType, dims) =>
  Buffer.concat([
    strField(1, name),
    lenField(2, lenField(1, Buffer.concat([intField(1, elemType), lenField(2, Buffer.concat(dims.map((d) => lenField(1, intField(1, d)))))]))),
  ]);
const f32 = (xs) => Buffer.from(new Float32Array(xs).buffer);
const i64 = (xs) => Buffer.from(new BigInt64Array(xs.map(BigInt)).buffer);

const NAMES = { 0: "person", 1: "bicycle", 2: "car" };
const BOXES_IN = [8, 8, 24, 40, 32, 12, 60, 52]; // xyxy in the 64x64 model input
const metaFor = (sha256) => ({
  schema_version: 1, task: "detect", model: "nitid1s", imgsz: S,
  input: { name: "images", shape: [1, 3, S, S], dtype: "float32", range: "0-1", color: "RGB", resize: "squash" },
  outputs: [{ name: "labels", dtype: "int64" }, { name: "boxes", dtype: "float32", format: "xyxy_input_px" }, { name: "scores", dtype: "float32" }],
  names: NAMES, topk: 2, default_iou: 0.85, nitid_version: "0.0.0", ...(sha256 ? { sha256 } : {}),
});
function buildOnnx() {
  const node = (inp, out) => Buffer.concat([strField(1, inp), strField(2, out), strField(4, "Identity")]);
  const graph = Buffer.concat([
    lenField(1, node("c_labels", "labels")),
    lenField(1, node("c_boxes", "boxes")),
    lenField(1, node("c_scores", "scores")),
    strField(2, "tiny"),
    lenField(5, tensor("c_labels", [1, 2], 7, i64([0, 2]))),
    lenField(5, tensor("c_boxes", [1, 2, 4], 1, f32(BOXES_IN))),
    lenField(5, tensor("c_scores", [1, 2], 1, f32([0.9, 0.6]))),
    lenField(11, valueInfo("images", 1, [1, 3, S, S])),
    lenField(12, valueInfo("labels", 7, [1, 2])),
    lenField(12, valueInfo("boxes", 1, [1, 2, 4])),
    lenField(12, valueInfo("scores", 1, [1, 2])),
  ]);
  const metaEntry = Buffer.concat([strField(1, "nitid_meta"), strField(2, JSON.stringify(metaFor()))]);
  return Buffer.concat([
    intField(1, 8),
    strField(2, "smoke"),
    lenField(7, graph),
    lenField(8, Buffer.concat([strField(1, ""), intField(2, 17)])),
    lenField(14, metaEntry),
  ]);
}
const ONNX = buildOnnx();
const SHA = createHash("sha256").update(ONNX).digest("hex");

/* ---------------- mock servers ---------------- */
const TOKEN = "smoke-token";
const json = (res, status, body, headers = {}) => {
  res.writeHead(status, { "Content-Type": "application/json", ...headers });
  res.end(JSON.stringify(body));
};
// What the browser's squash resize + box rescale gives for a 200x100 image.
const scaled = ([x1, y1, x2, y2]) => ({ x1: (x1 * 200) / S, y1: (y1 * 100) / S, x2: (x2 * 200) / S, y2: (y2 * 100) / S });
const det = (cls, conf, b) => ({ box: scaled(b), confidence: conf, class_id: cls, name: NAMES[cls], has_mask: false });
// Server segmentation result: the person carries an outline (triangle inside its 25,12.5-75,62.5 box).
const OUTLINE = { x: [25, 60, 25], y: [12.5, 12.5, 62.5] };
const RUN = {
  id: "r_1", model: "nitid1s", task: "segment", backend: "torch", device: "cpu", status: "done", error_msg: null, item_count: 1,
  items: [{
    id: 0, name: "white.png", width: 200, height: 100, speed: { preprocess: 3.1, inference: 28.4, postprocess: 1.2 },
    detections: [{ ...det(0, 0.91, BOXES_IN.slice(0, 4)), has_mask: true, segments: OUTLINE }, det(2, 0.61, BOXES_IN.slice(4))],
  }],
};
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm" };
// Keep in sync with CONTENT_SECURITY_POLICY in nitid/studio/security.py (the real server's policy).
const CSP =
  "default-src 'self'; worker-src 'self' blob:; script-src 'self' 'wasm-unsafe-eval'; " +
  "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'";

const stats = { onnxGets: 0, metaGets: 0, catalogOnnxGets: 0, unauthorized: 0, runPosts: 0 };

function startServer({ distDir, api }) {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname.startsWith("/api/v1/")) {
      if (!api) return json(res, 404, { error: { code: "not_found", message: "no api" } });
      const path = url.pathname.slice("/api/v1".length);
      if (path === "/health") return json(res, 200, { status: "ok", nitid_version: "0.0.0", api_version: 1 });
      if (req.headers.authorization !== `Bearer ${TOKEN}`) {
        stats.unauthorized += 1;
        return json(res, 401, { error: { code: "unauthorized", message: "Invalid token." } });
      }
      if (path === "/devices") return json(res, 200, { torch: { cpu: true, cuda: [] }, openvino: { available: true, devices: ["CPU"] } });
      if (path === "/models") return json(res, 200, [{ id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" }]);
      if (path === "/onnx/nitid1s/meta") {
        stats.metaGets += 1;
        return json(res, 200, metaFor(SHA));
      }
      if (path === "/onnx/nitid1s") {
        if (req.method === "GET") stats.onnxGets += 1;
        res.writeHead(200, { "Content-Type": "application/octet-stream", ETag: `"${SHA}"`, "Content-Length": ONNX.length });
        return res.end(req.method === "HEAD" ? undefined : ONNX);
      }
      if (path === "/runs" && req.method === "POST") {
        stats.runPosts += 1;
        await new Promise((done) => req.resume().on("end", done)); // drain the upload
        return json(res, 202, { id: "r_1" });
      }
      if (path === "/runs/r_1") return json(res, 200, RUN);
      if (path === "/runs/r_1/items/0/image") {
        res.writeHead(200, { "Content-Type": "image/png" });
        return res.end(PNG);
      }
      return json(res, 404, { error: { code: "not_found", message: "no such route" } });
    }
    if (url.pathname === "/catalog/models.json")
      return json(res, 200, { models: [{ id: "tiny", task: "detect", url: "tiny.onnx", sha256: SHA, label: "Tiny hosted" }] });
    if (url.pathname === "/catalog/tiny.onnx") {
      stats.catalogOnnxGets += 1;
      res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": ONNX.length });
      return res.end(ONNX);
    }
    let file = resolve(distDir, `.${normalize(url.pathname)}`);
    if (!file.startsWith(distDir + "/") || !existsSync(file) || file === distDir) file = join(distDir, "index.html");
    try {
      const body = await readFile(file);
      res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream", "Content-Security-Policy": CSP });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r({ server, base: `http://127.0.0.1:${server.address().port}` })));
}

/* ---------------- hosted build (with a model catalogue) ---------------- */
// SMOKE_HOSTED_DIR: reuse an earlier hosted build (it must have been built with STUDIO_MODELS_BASE_URL=/catalog/).
const hostedDist = process.env.SMOKE_HOSTED_DIR ?? mkdtempSync(join(tmpdir(), "studio-hosted-"));
if (!existsSync(join(hostedDist, "index.html"))) {
  console.log(`building the hosted variant into ${hostedDist} ...`);
  execFileSync("pnpm", ["run", "build:hosted"], {
    stdio: "ignore",
    env: { ...process.env, STUDIO_OUT_DIR: hostedDist, STUDIO_MODELS_BASE_URL: "/catalog/" },
  });
}

/* ---------------- test harness ---------------- */
let failed = false;
function check(name, ok, extra = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${extra ? ` (${extra})` : ""}`);
  if (!ok) failed = true;
}

const local = await startServer({ distDir: localDist, api: true });
const noApi = await startServer({ distDir: localDist, api: false });
const hosted = await startServer({ distDir: hostedDist, api: false });

const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome", args: ["--no-sandbox"] });
const errors = [];
const failedLoads = [];
async function newPage(context, { probe404 = false } = {}) {
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(`pageerror: ${e}`));
  page.on("response", (r) => r.status() >= 400 && failedLoads.push(`${r.status()} ${new URL(r.url()).pathname}`));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    // A default ("auto") build on a static host probes /api/v1/health and gets a 404: Chrome logs that
    // network failure itself. The app logs nothing; `build:hosted` does not probe at all.
    if (probe404 && /Failed to load resource.*404/.test(m.text())) return;
    errors.push(`console: ${m.text()}`);
  });
  return page;
}
const navLinks = (page) => page.getByRole("navigation", { name: "Main" }).getByRole("link").allTextContents();
const rows = (page) => page.locator("table.detections tbody tr").count();

try {
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await newPage(context);

  /* ---- 1. local deployment, server mode ---- */
  await page.goto(`${local.base}/#t=${TOKEN}`);
  await page.getByRole("button", { name: "Use local mode" }).waitFor();
  check("local: navigation has every entry", (await navLinks(page)).join(",") === "Home,Local,Browser,Compare,About", (await navLinks(page)).join(","));
  await page.keyboard.press("Tab");
  check("keyboard: the first Tab stop is the skip link", await page.evaluate(() => document.activeElement?.textContent === "Skip to content"));
  await page.keyboard.press("Tab");
  check("keyboard: the second Tab stop is in the navigation", await page.evaluate(() => document.activeElement?.closest("nav")?.getAttribute("aria-label") === "Main"));

  await page.getByRole("link", { name: "Local" }).click();
  await page.getByRole("heading", { name: "Local mode" }).waitFor();
  await page.getByLabel("Model and task").waitFor();
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 15000 });
  check("server mode: 2 detections drawn", (await rows(page)) === 2);
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas.detection-canvas");
    const d = c.getContext("2d").getImageData(90, 80, 1, 1).data;
    return d[0] === 255 && d[1] === 255 && d[2] === 255;
  });
  const serverPx = await page.locator("canvas.detection-canvas").evaluate((c) => {
    const px = (x, y) => Array.from(c.getContext("2d").getImageData(x, y, 1, 1).data);
    return { inTriangle: px(32, 20), outTriangle: px(70, 55) };
  });
  const tinted = (p) => p[0] < 250 || p[1] < 250 || p[2] < 250;
  check("server mode: outline interior is tinted", tinted(serverPx.inTriangle), JSON.stringify(serverPx.inTriangle));
  check("server mode: box area outside the outline is untinted", !tinted(serverPx.outTriangle), JSON.stringify(serverPx.outTriangle));
  check("server mode: no stale 'outlines not provided' note", (await page.getByText(/Mask outlines are not provided/).count()) === 0);
  await page.getByRole("button", { name: "Export" }).click();
  const [jsonServer] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download JSON" }).click()]);
  const serverDoc = JSON.parse(readFileSync(await jsonServer.path(), "utf8"));
  check("server mode: JSON export", serverDoc.source === "server" && serverDoc.detection_count === 2 && serverDoc.runtime.backend === "torch", jsonServer.suggestedFilename());
  check("server mode: JSON carries the outline, no pixel count", Array.isArray(serverDoc.detections[0].mask?.polygon) && serverDoc.detections[0].mask.polygon.length === 3 && !("pixels" in serverDoc.detections[0].mask) && serverDoc.detections[1].mask === undefined, JSON.stringify(serverDoc.detections[0].mask));
  const [yoloServer] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download YOLO TXT" }).click()]);
  // Two files (labels + classes.txt) arrive as a ZIP; read the labels with the system unzip.
  const yoloLines = execFileSync("unzip", ["-p", await yoloServer.path(), "white.txt"], { encoding: "utf8" }).trim().split("\n");
  check("server mode: YOLO TXT has a polygon line for the outline and a box line for the rest", yoloLines.length === 2 && yoloLines[0].split(" ").length === 7 && yoloLines[1].split(" ").length === 5, yoloLines.join(" | "));
  check("server mode: PDF export is enabled (image loaded)", await page.getByRole("button", { name: "Download PDF report" }).isEnabled());

  /* ---- 2. in-browser mode, real ONNX Runtime Web ---- */
  await page.getByRole("link", { name: "Browser" }).click();
  await page.getByRole("heading", { name: "Browser mode" }).waitFor();
  await page.getByLabel("Model and task").waitFor();
  check("browser mode: lists the server's model", (await page.getByLabel("Model and task").locator("option").allTextContents()).join() === "nitid1s (detect)");
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
  check("browser mode: model downloaded once, with the token", stats.onnxGets === 1 && stats.unauthorized === 0, `gets=${stats.onnxGets} 401s=${stats.unauthorized}`);
  check("browser mode: 2 detections from the real session", (await rows(page)) === 2);
  const canvas = page.getByRole("img", { name: /white\.png: 2 detections shown/ });
  await canvas.waitFor();
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas.detection-canvas");
    const d = c.getContext("2d").getImageData(90, 50, 1, 1).data;
    return d[0] === 255 && d[1] === 255 && d[2] === 255;
  });
  const edge = await canvas.evaluate((c) => Array.from(c.getContext("2d").getImageData(25, 40, 2, 1).data));
  check("browser mode: box edge is drawn on the image", edge.slice(0, 3).some((v) => v < 200) || edge.slice(4, 7).some((v) => v < 200), JSON.stringify(edge));
  await page.getByRole("button", { name: "Export" }).click();
  const [jsonBrowser] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download JSON" }).click()]);
  const browserDoc = JSON.parse(readFileSync(await jsonBrowser.path(), "utf8"));
  check("browser mode: JSON export from the engine mapper", browserDoc.source === "browser" && browserDoc.detection_count === 2 && browserDoc.runtime.backend === "wasm");
  const box = browserDoc.detections[0].box;
  check("browser mode: box rescaled to the original image", Math.abs(box.x1 - 25) < 0.5 && Math.abs(box.y2 - 62.5) < 0.5, JSON.stringify(box));
  const slider = page.getByRole("slider", { name: /Minimum confidence/ });
  await slider.focus();
  await slider.fill("75");
  check("browser mode: slider hides the 0.6 car", (await rows(page)) === 1);

  await page.reload();
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
  check("browser mode: after a reload the model comes from the Cache API", stats.onnxGets === 1, `gets=${stats.onnxGets}`);

  /* ---- 3. comparison ---- */
  await page.getByRole("link", { name: "Compare" }).click();
  await page.getByRole("heading", { name: "Compare", level: 1 }).waitFor();
  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Compare" }).focus();
  await page.keyboard.press("Enter");
  await page.getByTestId("agreement").waitFor({ timeout: 60000 });
  const agreement = await page.getByTestId("agreement").textContent();
  check("compare: server and browser agree on both boxes", /Agreement 100%/.test(agreement) && /2 matched/.test(agreement), agreement.slice(0, 80));
  check("compare: both sides have a canvas, stats and an export menu", (await page.locator("canvas.detection-canvas").count()) === 2 && (await page.getByRole("button", { name: "Export" }).count()) === 2);
  check("compare: timings shown per side", /Inference/.test(await page.getByTestId("A-stats").textContent()) && /model load/.test(await page.getByTestId("B-stats").textContent()));

  /* ---- 4a. a static host with no server and no catalogue ---- */
  const bare = await newPage(await browser.newContext(), { probe404: true });
  await bare.goto(`${noApi.base}/`);
  await bare.getByRole("button", { name: "Use browser mode" }).waitFor();
  check("hosted (default build): no server-only navigation", (await navLinks(bare)).join(",") === "Home,Browser,About", (await navLinks(bare)).join(","));
  await bare.getByRole("link", { name: "Browser" }).click();
  await bare.getByTestId("no-models").waitFor();
  check("hosted (default build): clear 'no models configured' state", /No models are configured/.test(await bare.getByTestId("no-models").textContent()));
  await bare.goto(`${noApi.base}/compare`);
  const needServer = bare.getByRole("alert").filter({ hasText: "nitid studio" });
  await needServer.waitFor();
  check("hosted: /compare explains that a server is needed", await needServer.isVisible());

  /* ---- 4b. hosted build with a catalogue ---- */
  const host = await newPage(await browser.newContext());
  await host.goto(`${hosted.base}/run/browser`);
  await host.getByRole("option", { name: "Tiny hosted" }).waitFor({ state: "attached", timeout: 15000 });
  await host.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await host.getByRole("button", { name: "Run" }).click();
  await host.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60000 });
  check("hosted (catalogue): model from the configured base, 2 detections", stats.catalogOnnxGets === 1 && (await rows(host)) === 2, `gets=${stats.catalogOnnxGets}`);
  check("hosted (catalogue): export menu present", await host.getByRole("button", { name: "Export" }).isVisible());

  check("no console or page errors anywhere", errors.length === 0, errors.join(" | "));
  check("the only failed requests are the health probes of the static host", failedLoads.every((f) => f === "404 /api/v1/health"), failedLoads.join(" | "));
} finally {
  await browser.close();
  local.server.close();
  noApi.server.close();
  hosted.server.close();
}
process.exit(failed ? 1 : 0);
