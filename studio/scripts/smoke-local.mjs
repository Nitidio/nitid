/* global document */
// Browser smoke test of the built Local panel against a mocked API (no Python backend needed).
// Usage (after `pnpm run build`):  node scripts/smoke-local.mjs
// Needs playwright-core (resolved from PLAYWRIGHT_CORE_PATH, or common global locations) and
// a Chrome/Chromium (CHROME_PATH, default /usr/bin/google-chrome). Not part of `pnpm run check`.
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
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

const distDir = resolve(process.env.STUDIO_OUT_DIR ?? "../nitid/studio/static");
if (!existsSync(join(distDir, "index.html"))) throw new Error(`Build first: ${distDir}/index.html missing`);

/* ---- a 200x100 white PNG ---- */
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
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // RGB
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

/* ---- mock API ---- */
const TOKEN = "smoke-token";
const seenAuth = [];
let runPolls = 0;
let uploaded = null;
const json = (res, status, body) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
};
const detection = (cls, name, conf, box) => ({ box, confidence: conf, class_id: cls, name, has_mask: false });
const RUN = {
  id: "r_1", model: "nitid1s", task: "detect", backend: "torch", device: "cpu", status: "done",
  error_msg: null, item_count: 1,
  items: [{
    id: 0, name: "white.png", width: 200, height: 100,
    speed: { preprocess: 3.1, inference: 28.4, postprocess: 1.2 },
    detections: [
      detection(0, "person", 0.91, { x1: 20, y1: 10, x2: 80, y2: 90 }),
      detection(2, "car", 0.4, { x1: 100, y1: 20, x2: 190, y2: 80 }),
    ],
  }],
};

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname.startsWith("/api/v1/")) {
    const path = url.pathname.slice("/api/v1".length);
    if (path !== "/health") {
      seenAuth.push(req.headers.authorization ?? null);
      if (req.headers.authorization !== `Bearer ${TOKEN}`)
        return json(res, 401, { error: { code: "unauthorized", message: "Invalid token." } });
    }
    if (path === "/health") return json(res, 200, { status: "ok", nitid_version: "0.0.0", api_version: "1" });
    if (path === "/devices")
      return json(res, 200, { torch: { cpu: true, cuda: [] }, openvino: { available: true, devices: ["CPU", "GPU"] } });
    if (path === "/models")
      return json(res, 200, [{ id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" }]);
    if (path === "/runs" && req.method === "POST") {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      uploaded = Buffer.concat(chunks).toString("latin1");
      return json(res, 202, { id: "r_1" });
    }
    if (path === "/runs/r_1") {
      runPolls += 1;
      return json(res, 200, runPolls < 2 ? { ...RUN, status: "running", items: [] } : RUN);
    }
    if (path === "/runs/r_1/items/0/image") {
      res.writeHead(200, { "Content-Type": "image/png" });
      return res.end(PNG);
    }
    return json(res, 404, { error: { code: "not_found", message: "no such route" } });
  }
  let file = resolve(distDir, `.${normalize(url.pathname)}`);
  if (!file.startsWith(distDir + "/") || !existsSync(file)) file = join(distDir, "index.html");
  try {
    const body = await readFile(file);
    res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

let failed = false;
function check(name, ok, extra = "") {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${extra ? ` (${extra})` : ""}`);
  if (!ok) failed = true;
}

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome",
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  await page.goto(`${base}/#t=${TOKEN}`);
  await page.getByRole("button", { name: "Use local mode" }).click();
  await page.getByLabel("Model and task").waitFor();
  check("panel loads models with the bearer token", seenAuth.length > 0 && seenAuth.every((a) => a === `Bearer ${TOKEN}`));
  check("run disabled without images", await page.getByRole("button", { name: "Run" }).isDisabled());

  await page.getByLabel("Images").setInputFiles({ name: "white.png", mimeType: "image/png", buffer: PNG });
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 15000 });
  check("multipart carries files and params", uploaded?.includes('name="files"') && uploaded.includes('"model":"nitid1s"'));

  const canvas = page.getByRole("img", { name: /white\.png: 2 detections shown/ });
  await canvas.waitFor();
  // Wait for the original image to be drawn, then sample: white background, coloured box edge.
  const sample = () =>
    canvas.evaluate((c) => {
      const ctx = c.getContext("2d");
      const px = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data);
      return { inside: px(50, 50), edge: px(20, 50) };
    });
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas.detection-canvas");
    const d = c.getContext("2d").getImageData(50, 50, 1, 1).data;
    return d[0] === 255 && d[1] === 255 && d[2] === 255;
  });
  const px = await sample();
  check("original image drawn", px.inside[0] === 255 && px.inside[3] === 255, JSON.stringify(px.inside));
  check("box edge is coloured (not white)", px.edge[0] < 200 || px.edge[1] < 200 || px.edge[2] < 200, JSON.stringify(px.edge));

  const rows = () => page.locator("table.detections tbody tr").count();
  check("two rows at 25%", (await rows()) === 2);
  const slider = page.getByRole("slider", { name: /Minimum confidence/ });
  await slider.focus();
  await slider.fill("60");
  check("slider hides the 40% car", (await rows()) === 1);
  await page.getByRole("checkbox", { name: /^person/ }).uncheck();
  check("class filter hides person", (await page.getByText("Nothing above the current threshold.").count()) === 1);

  check("no console/page errors", errors.length === 0, errors.join(" | "));
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
