/* global window */
// Browser half of the server-vs-browser parity test: loads the engine harness served by the real
// Studio app (same CSP/COOP/COEP headers), downloads the ONNX from the real /api/v1/onnx endpoint,
// runs the engine in real Chrome and writes detections (with masks) to the result file.
//   node parity.mjs --url http://127.0.0.1:PORT/#t=TOKEN --out result.json
// The Python test compares `data.runs` with the server's /runs output (see test_studio_parity.py).
import { launchChrome, parseArgs, Recorder, splitStudioUrl, watchPage } from "./lib.mjs";

const args = parseArgs();
const { origin, token } = splitStudioUrl(args.url);
const rec = new Recorder(args.out);
const CONFS = [0.5, 0.1, 0.9];
const CASES = [
  { key: "detect", model: "tiny-det", task: "detect" },
  { key: "segment", model: "tiny-seg", task: "segment" },
];

/** Chrome switches tried to get a WebGPU adapter in headless mode (observation only). */
const GPU_VARIANTS = {
  default: [],
  "unsafe-webgpu": ["--enable-unsafe-webgpu"],
  "unsafe-webgpu+swiftshader": ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"],
};

async function openHarness(browser) {
  const page = await browser.newPage();
  const watch = watchPage(page);
  await page.goto(`${origin}/`);
  await page.waitForFunction(() => window.__parity !== undefined, undefined, { timeout: 15_000 });
  return { page, watch };
}

const browser = await launchChrome();
try {
  const { page, watch } = await openHarness(browser);
  rec.data.environment = await page.evaluate(() => window.__parity.environment());
  const env = rec.data.environment;
  rec.check("harness page is cross-origin isolated under the Studio headers", env.crossOriginIsolated === true);
  rec.check("SharedArrayBuffer is available", env.sharedArrayBuffer === "function");

  rec.data.runs = {};
  for (const testCase of CASES) {
    const run = await page.evaluate(
      (a) => window.__parity.run(a),
      { token, model: testCase.model, task: testCase.task, imageUrl: "/image.png", confs: CONFS },
    );
    rec.data.runs[testCase.key] = run;
    rec.check(`${testCase.key}: ONNX downloaded from /api/v1/onnx and verified against its ETag`, Boolean(run.etag) && run.info.modelSource === "bytes");
    rec.check(`${testCase.key}: wasm backend started`, run.info.backend === "wasm", JSON.stringify(run.info));
    rec.check(`${testCase.key}: repeated run on the same session is deterministic`, run.repeatCount === run.results[String(CONFS[0])].detections.length);
  }

  rec.check("no console errors or failed requests while the engine ran", watch.unexpected(["http 401"]).length === 0, watch.unexpected(["http 401"]).join(" | "));
  // Known gap probe: the worker's own download (modelUrl) cannot carry the bearer token.
  const viaUrl = await page
    .evaluate((a) => window.__parity.run(a), { token, model: "tiny-det", task: "detect", imageUrl: "/image.png", confs: [0.5], viaWorkerUrl: true })
    .then(() => "worked")
    .catch((error) => String(error.message ?? error));
  rec.data.workerDownloadWithoutToken = viaUrl;

  await page.close();

  // WebGPU / isolation probes (observations; they never fail the run)
  rec.data.webgpuProbe = {};
  for (const [name, flags] of Object.entries(GPU_VARIANTS)) {
    const probeBrowser = await launchChrome(flags);
    try {
      const probe = await openHarness(probeBrowser);
      rec.data.webgpuProbe[name] = await probe.page.evaluate(() => window.__parity.environment());
    } finally {
      await probeBrowser.close();
    }
  }
} catch (error) {
  rec.check("driver completed without an exception", false, String(error?.stack ?? error));
} finally {
  await browser.close();
}
rec.finish();
