/* global document, sessionStorage, window */
// End-to-end drive of the built Studio UI against a real `nitid studio` process.
//   node ui.mjs --url http://127.0.0.1:PORT/#t=TOKEN --image image.png --out result.json
// Covers: tokenised URL, local-mode detection, model picker (local *.pth files), upload, run via
// server mode, boxes and table, confidence filter, second run with another model, the server
// queue, and the error state for a wrong and a missing token. Needs fixtures from make_fixtures.py.
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { launchChrome, parseArgs, Recorder, splitStudioUrl, watchPage } from "./lib.mjs";

const args = parseArgs();
const { origin, token, url } = splitStudioUrl(args.url);
const imageBytes = readFileSync(args.image);
const imageName = basename(args.image);
const rec = new Recorder(args.out);
const auth = { Authorization: `Bearer ${token}` };

async function api(path, init = {}) {
  const response = await fetch(`${origin}/api/v1${path}`, { ...init, headers: { ...auth, ...init.headers } });
  return { status: response.status, json: await response.json().catch(() => null) };
}

/** Submit a run directly to the API (the independent oracle for what the UI shows). */
async function submitRun(model, task, conf = 0.1) {
  const form = new FormData();
  form.append("files", new Blob([imageBytes], { type: "image/png" }), imageName);
  form.append("params", JSON.stringify({ model, task, backend: "torch", device: "cpu", conf }));
  const response = await fetch(`${origin}/api/v1/runs`, { method: "POST", headers: auth, body: form });
  return { status: response.status, id: (await response.json()).id };
}

async function waitDone(id, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const { json } = await api(`/runs/${id}`);
    if (json.status === "done" || json.status === "failed") return json;
    if (Date.now() > deadline) throw new Error(`run ${id} did not finish`);
    await new Promise((r) => setTimeout(r, 50));
  }
}

const roundedBox = (b) => `${Math.round(b.x1)}, ${Math.round(b.y1)}, ${Math.round(b.x2)}, ${Math.round(b.y2)}`;

const browser = await launchChrome();
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const watch = watchPage(page);

  // ---- 1. tokenised URL and local-mode detection
  await page.goto(url);
  await page.getByText("Connected to a local server.").waitFor({ timeout: 15_000 });
  rec.check("local mode detected from /health", true);
  rec.check("token fragment removed from the address bar", (await page.evaluate(() => window.location.hash)) === "");
  rec.check("token kept for this tab only (sessionStorage)", (await page.evaluate(() => sessionStorage.length)) === 1);
  rec.check("local mode card enabled", await page.getByRole("button", { name: "Use local mode" }).isEnabled());

  // ---- 2. model picker shows the local *.pth files
  await page.getByRole("button", { name: "Use local mode" }).click();
  const picker = page.getByLabel("Model and task");
  await picker.waitFor();
  const options = await picker.locator("option").allTextContents();
  rec.data.modelOptions = options;
  rec.check("local detect checkpoint listed", options.some((o) => o.includes("tiny-det") && o.includes("[local file]")));
  rec.check("local segment checkpoint listed", options.some((o) => o.includes("tiny-seg") && o.includes("[local file]")));
  rec.check(
    "registry models that are not downloaded are marked",
    options.some((o) => o.startsWith("nitid1s") && o.endsWith("not downloaded")),
  );
  rec.check("downloaded local detect model preselected", (await picker.inputValue()) === "detect:tiny-det");
  rec.check("model reported ready", await page.getByText("Ready on this computer.").isVisible());
  rec.check("Run disabled without an image", await page.getByRole("button", { name: "Run" }).isDisabled());
  rec.check("backend defaults to torch", (await page.getByLabel("Backend").inputValue()) === "torch");
  const backendOptions = await page.getByLabel("Backend").locator("option").allTextContents();
  rec.data.backendOptions = backendOptions;

  // ---- 3. upload and run through the server
  await page.getByLabel("Images").setInputFiles({ name: imageName, mimeType: "image/png", buffer: imageBytes });
  rec.check("Run enabled once an image is chosen", await page.getByRole("button", { name: "Run" }).isEnabled());
  const oracle1 = await (async () => {
    const run = await submitRun("tiny-det", "detect");
    return waitDone(run.id);
  })();
  rec.check("oracle run finished", oracle1.status === "done", oracle1.error_msg ?? "");
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 90_000 });
  rec.check("results header names model, task, backend and device", await page.getByText(/tiny-det \(detect\) on torch \/ cpu/).isVisible());

  const detections1 = oracle1.items[0].detections;
  const above = (list, t) => list.filter((d) => d.confidence >= t);
  const canvas = page.locator("canvas.detection-canvas");
  const label1 = await canvas.getAttribute("aria-label");
  rec.check(
    "canvas names the image and the number of detections",
    label1 === `${imageName}: ${above(detections1, 0.25).length} detections shown`,
    String(label1),
  );
  const rows = page.locator("table.detections tbody tr");
  rec.check("table rows equal API detections at 25%", (await rows.count()) === above(detections1, 0.25).length, `${await rows.count()} rows`);
  const firstRow = (await rows.first().locator("td").allTextContents()).map((s) => s.trim());
  const expectedFirst = above(detections1, 0.25)[0];
  rec.check(
    "first row matches the API (class, confidence, rounded box)",
    firstRow[0] === expectedFirst.name &&
      firstRow[1] === `${(expectedFirst.confidence * 100).toFixed(1)}%` &&
      firstRow[2] === roundedBox(expectedFirst.box),
    JSON.stringify(firstRow),
  );
  // The canvas must hold the decoded image (not stay blank) and have its size.
  await page.waitForFunction(() => {
    const c = document.querySelector("canvas.detection-canvas");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    const seen = new Set();
    for (let i = 0; i < d.length && seen.size < 4; i += 4 * 97) seen.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
    return seen.size >= 4;
  });
  const size = await canvas.evaluate((c) => [c.width, c.height]);
  rec.check("canvas drawn at the original image size", size[0] === 320 && size[1] === 240, size.join("x"));
  rec.data.timingLine = (await page.locator("p.hint[role=status]").first().textContent())?.trim();

  // ---- 4. client-side confidence filter
  const slider = page.getByRole("slider", { name: /Minimum confidence/ });
  await slider.focus();
  await slider.fill("50");
  rec.check("slider to 50% matches the API", (await rows.count()) === above(detections1, 0.5).length, `${await rows.count()} rows`);
  await slider.fill("95");
  rec.check("slider to 95% shows the empty message", await page.getByText("Nothing above the current threshold.").isVisible());

  // ---- 5. second run, other model (sequential runs reuse the server queue)
  await picker.selectOption({ label: options.find((o) => o.includes("tiny-seg")) });
  const oracle2 = await waitDone((await submitRun("tiny-seg", "segment")).id);
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByText(/tiny-seg \(segment\) on torch/).waitFor({ timeout: 90_000 });
  rec.check("second run replaced the results (segment)", true);
  rec.check("mask column appears for segment results", await page.getByRole("columnheader", { name: "Mask" }).isVisible());
  const rows2 = await page.locator("table.detections tbody tr").count();
  rec.check("segment table rows equal API detections", rows2 === above(oracle2.items[0].detections, 0.25).length, `${rows2} rows`);
  rec.data.staleMaskNote = await page.getByText(/Mask outlines are not provided/).isVisible();

  // ---- 6. the server queue: three runs submitted at once finish in order; the UI run waits behind them
  const t0 = Date.now();
  const queued = [];
  for (let i = 0; i < 3; i++) queued.push((await submitRun("tiny-seg", "segment")).id);
  const states = await Promise.all(queued.map((id) => api(`/runs/${id}`).then((r) => r.json.status)));
  rec.check("a run submitted behind another is pending", states.includes("pending"), states.join(","));
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByText("Queued on the server...").waitFor({ timeout: 20_000 });
  rec.check("UI shows the queued state while the server is busy", true);
  const finishOrder = [];
  const pendingIds = new Set(queued);
  while (pendingIds.size > 0 && Date.now() - t0 < 120_000) {
    for (const id of [...pendingIds]) {
      const { json } = await api(`/runs/${id}`);
      if (json.status === "done") {
        finishOrder.push(id);
        pendingIds.delete(id);
      }
    }
    await new Promise((r) => setTimeout(r, 20));
  }
  rec.check("queued runs finish in submission order", JSON.stringify(finishOrder) === JSON.stringify(queued), finishOrder.join(","));
  await page.getByRole("heading", { name: "Results" }).waitFor({ timeout: 60_000 });
  await page.getByText("Queued on the server...").waitFor({ state: "detached", timeout: 60_000 });
  rec.check("UI left the queued state and shows results", await page.getByRole("heading", { name: "Results" }).isVisible());

  // ---- 7. a reload of the same tab keeps working (token mirrored in sessionStorage)
  await page.goto(`${origin}/run/local`);
  await page.getByLabel("Model and task").waitFor({ timeout: 15_000 });
  rec.check("deep link /run/local works after a reload", true);

  const unexpected = watch.unexpected();
  rec.check("no console errors, page errors or failed requests (main flow)", unexpected.length === 0, unexpected.join(" | "));
  await context.close();

  // ---- 8. error state: wrong token and missing token
  for (const [name, entry] of [
    ["wrong token", `${origin}/#t=this-is-not-the-token-0000`],
    ["missing token", `${origin}/`],
  ]) {
    const bad = await browser.newContext();
    const badPage = await bad.newPage();
    const badWatch = watchPage(badPage);
    await badPage.goto(entry);
    await badPage.getByText("Connected to a local server.").waitFor({ timeout: 15_000 });
    rec.check(`${name}: health is public, mode still detected`, true);
    await badPage.getByRole("button", { name: "Use local mode" }).click();
    const alert = badPage.getByRole("alert").filter({ hasText: "Could not talk to the local server." });
    await alert.waitFor({ timeout: 15_000 });
    const text = (await alert.textContent()) ?? "";
    rec.data[`error ${name}`] = text.trim();
    rec.check(`${name}: error state explains the token`, /token/i.test(text) && text.includes("nitid studio"), text.trim());
    rec.check(`${name}: no model list rendered`, (await badPage.getByLabel("Model and task").count()) === 0);
    rec.check(`${name}: retry button offered`, await badPage.getByRole("button", { name: "Try again" }).isVisible());
    const other = badWatch.unexpected(["401"]);
    rec.check(`${name}: only the expected 401 errors reach the console`, other.length === 0, other.join(" | "));
    await bad.close();
  }
} catch (error) {
  rec.check("driver completed without an exception", false, String(error?.stack ?? error));
} finally {
  await browser.close();
}
rec.finish();
