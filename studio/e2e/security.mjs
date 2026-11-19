/* global document, window, fetch, Image, Worker, Blob, URL, WebAssembly, FormData, EvalError, TypeError */
// Security checks of a real `nitid studio` process in real Chrome.
//   node security.mjs --url http://127.0.0.1:PORT/#t=TOKEN --image image.png --out result.json
// Host guard (DNS rebinding emulated with a Chrome host-resolver rule), Origin guard (a hostile page
// on another origin), token required, CORS/CORP isolation, security headers on every response, CSP
// (inline script, eval, inline handler, remote script/fetch/image/frame, data: worker), no console errors.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { launchChrome, parseArgs, Recorder, splitStudioUrl, watchPage } from "./lib.mjs";

const args = parseArgs();
const { origin, port, token, url } = splitStudioUrl(args.url);
const rec = new Recorder(args.out);
const imageBytes = readFileSync(args.image);
const authHeader = { Authorization: `Bearer ${token}` };
const countRuns = async () => (await (await fetch(`${origin}/api/v1/runs`, { headers: authHeader })).json()).length;

// A hostile site on another origin (http://evil.test:EPORT, resolved to loopback by Chrome) that
// records every request it receives, so "blocked" can be proven by "never arrived".
const evilRequests = [];
const evil = createServer((req, res) => {
  evilRequests.push(`${req.method} ${req.url}`);
  res.writeHead(200, { "Content-Type": req.url.endsWith(".js") ? "text/javascript" : "text/html" });
  res.end(req.url.endsWith(".js") ? "window.__remoteScriptRan = true;" : "<!doctype html><title>evil</title>");
});
await new Promise((resolve) => evil.listen(0, "127.0.0.1", resolve));
const evilOrigin = `http://evil.test:${evil.address().port}`;

const REQUIRED_HEADERS = {
  "x-content-type-options": "nosniff",
  "cross-origin-opener-policy": "same-origin",
  "cross-origin-embedder-policy": "require-corp",
  "cross-origin-resource-policy": "same-origin",
  "referrer-policy": "no-referrer",
};
const hasSecurityHeaders = (headers) =>
  Object.entries(REQUIRED_HEADERS).every(([k, v]) => headers[k] === v) &&
  /script-src 'self' 'wasm-unsafe-eval'/.test(headers["content-security-policy"] ?? "");

const browser = await launchChrome([`--host-resolver-rules=MAP evil.test 127.0.0.1`]);
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const watch = watchPage(page);
  const responses = [];
  const requestUrls = [];
  const referers = [];
  page.on("response", async (r) => responses.push({ url: r.url(), status: r.status(), headers: r.headers() }));
  page.on("request", async (r) => {
    requestUrls.push(r.url());
    referers.push(r.headers()["referer"]);
  });

  // ---- security headers on the page and its assets, and the token's life in the browser
  await page.goto(url);
  await page.getByText("Connected to a local server.").waitFor({ timeout: 15_000 });
  await page.getByRole("button", { name: "Use local mode" }).click();
  await page.getByLabel("Model and task").waitFor();
  const own = responses.filter((r) => r.url.startsWith(origin));
  const kinds = new Set(own.map((r) => (r.url.includes("/api/") ? "api" : r.url.includes("/assets/") ? "asset" : "page")));
  rec.check("page, assets and API all loaded", ["page", "asset", "api"].every((k) => kinds.has(k)), [...kinds].join(","));
  const missing = own.filter((r) => !hasSecurityHeaders(r.headers)).map((r) => r.url);
  rec.check("every response carries nosniff, COOP, COEP, CORP, no-referrer and the CSP", missing.length === 0, missing.join(" "));
  rec.check("no token in any request URL", requestUrls.every((u) => !u.includes(token)));
  rec.check("no Referer header is sent", referers.every((r) => !r));
  rec.check("page is cross-origin isolated (COOP+COEP effective)", await page.evaluate(() => window.crossOriginIsolated === true));
  rec.check("SharedArrayBuffer available in the page", await page.evaluate(() => typeof SharedArrayBuffer === "function"));
  rec.check("no console errors on the main flow", watch.unexpected().length === 0, watch.unexpected().join(" | "));

  // ---- token required (navigations carry no Authorization header)
  const probe = await context.newPage();
  const status = async (path) => {
    const response = await probe.goto(`${origin}${path}`);
    return { status: response.status(), body: await response.json().catch(() => null), headers: response.headers() };
  };
  const health = await status("/api/v1/health");
  rec.check("/health is public", health.status === 200 && health.body.status === "ok");
  for (const path of ["/api/v1/models", "/api/v1/devices", "/api/v1/runs", "/api/v1/onnx/tiny-det?task=detect", "/api/v1/no-such-endpoint"]) {
    const r = await status(path);
    rec.check(
      `${path} without a token: 401 unauthorized envelope`,
      r.status === 401 && r.body?.error?.code === "unauthorized" && r.headers["www-authenticate"] === "Bearer" && hasSecurityHeaders(r.headers),
      `${r.status} ${JSON.stringify(r.body)}`,
    );
  }
  const viaPage = await page.evaluate(async () => {
    const get = async (headers) => (await fetch("/api/v1/models", { headers })).status;
    return { none: await get({}), wrong: await get({ Authorization: "Bearer wrong" }), basic: await get({ Authorization: "Basic abc" }) };
  });
  rec.check("same-origin fetch: no token, wrong token and wrong scheme are all 401", Object.values(viaPage).every((s) => s === 401), JSON.stringify(viaPage));
  const withToken = await page.evaluate(async (t) => (await fetch("/api/v1/models", { headers: { Authorization: `Bearer ${t}` } })).status, token);
  rec.check("same-origin fetch with the token: 200", withToken === 200);

  // ---- Host guard: DNS rebinding emulated (evil.test resolves to 127.0.0.1 but keeps its Host)
  const rebound = await probe.goto(`http://evil.test:${port}/api/v1/health`);
  const reboundBody = await rebound.json().catch(() => null);
  rec.check("rebinding: foreign Host on /health is 403 forbidden_host", rebound.status() === 403 && reboundBody?.error?.code === "forbidden_host", `${rebound.status()} ${JSON.stringify(reboundBody)}`);
  const reboundPage = await probe.goto(`http://evil.test:${port}/`);
  rec.check("rebinding: the UI is not served to a foreign Host", reboundPage.status() === 403, String(reboundPage.status()));
  rec.check("rebinding: the rejection still has the security headers", hasSecurityHeaders(reboundPage.headers()));
  const localhostPage = await probe.goto(`http://localhost:${port}/`);
  rec.check("Host localhost:PORT is allowed (loopback name)", localhostPage.status() === 200);
  const trailing = await probe.goto(`http://127.0.0.1:${port}/%2e%2e/%2e%2e/etc/passwd`);
  rec.check("path traversal is not served as a file", trailing.status() !== 200 || (await probe.content()).includes("<div id=\"root\">"), String(trailing.status()));

  // ---- Origin guard + CORS/CORP: a hostile page on another origin talks to Studio
  const attacker = await context.newPage();
  const attackerSeen = [];
  attacker.on("response", (r) => r.url().startsWith(origin) && attackerSeen.push({ url: r.url(), status: r.status(), method: r.request().method() }));
  attacker.on("request", (r) => r.url().startsWith(origin) && attackerSeen.push({ url: r.url(), sent: true, method: r.method(), auth: r.headers()["authorization"] }));
  const attackerCdp = await context.newCDPSession(attacker);
  await attackerCdp.send("Network.enable");
  const blocked = [];
  attackerCdp.on("Network.loadingFailed", (e) => blocked.push(`${e.errorText} ${e.blockedReason ?? ""}`.trim()));
  await attacker.goto(`${evilOrigin}/`);
  const runsBefore = await countRuns();
  const attack = await attacker.evaluate(async ({ target, tok, png }) => {
    const out = {};
    try {
      await fetch(`${target}/api/v1/health`);
      out.corsRead = "allowed";
    } catch (error) {
      out.corsRead = `blocked:${error.name}`;
    }
    try {
      await fetch(`${target}/api/v1/runs`, { method: "POST", mode: "no-cors", body: new FormData() });
      out.noCorsPost = "response readable";
    } catch (error) {
      out.noCorsPost = `response blocked:${error.name}`;
    }
    try {
      // The worst case: the page somehow holds a valid token and a valid upload.
      const form = new FormData();
      form.append("files", new Blob([new Uint8Array(png)], { type: "image/png" }), "x.png");
      form.append("params", JSON.stringify({ model: "tiny-det", task: "detect", backend: "torch", device: "cpu" }));
      await fetch(`${target}/api/v1/runs`, { method: "POST", headers: { Authorization: `Bearer ${tok}` }, body: form });
      out.corsPost = "allowed";
    } catch (error) {
      out.corsPost = `blocked:${error.name}`;
    }
    out.img = await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve("loaded");
      img.onerror = () => resolve("blocked");
      img.src = `${target}/api/v1/health`;
    });
    return out;
  }, { target: origin, tok: token, png: [...imageBytes] });
  await new Promise((resolve) => setTimeout(resolve, 500));
  const runsAfter = await countRuns();
  rec.data.attack = attack;
  rec.data.attackBlockedReasons = blocked;
  rec.check("hostile page cannot read /health cross-origin (no CORS headers)", attack.corsRead.startsWith("blocked"), attack.corsRead);
  rec.check(
    "cross-origin POST with a valid token and upload is rejected by the browser and creates no run",
    attack.corsPost.startsWith("blocked") && runsAfter === runsBefore,
    `${attack.corsPost}; runs ${runsBefore} -> ${runsAfter}`,
  );
  rec.check("no-cors POST: response is unreadable for the hostile page (CORP)", attack.noCorsPost.startsWith("response blocked") && blocked.some((b) => b.includes("NotSameOrigin")), `${attack.noCorsPost}; ${blocked.join(",")}`);
  rec.check("hostile page cannot embed an API response as an image (CORP)", attack.img === "blocked", attack.img);

  // A classic CSRF: a hostile page submits a form to Studio (top-level navigation, Origin = hostile page).
  await attacker.goto(`${evilOrigin}/`);
  await attacker.evaluate((target) => {
    const form = document.createElement("form");
    form.method = "POST";
    form.action = `${target}/api/v1/runs`;
    form.enctype = "multipart/form-data";
    const field = document.createElement("input");
    field.name = "params";
    field.value = "{}";
    form.append(field);
    document.body.append(form);
    form.submit();
  }, origin);
  await attacker.waitForURL(`${origin}/api/v1/runs`, { timeout: 10_000 });
  const csrfBody = JSON.parse(await attacker.locator("body").innerText());
  rec.check("CSRF form POST from a foreign Origin: 403 forbidden_origin (even before the token check)", csrfBody?.error?.code === "forbidden_origin", JSON.stringify(csrfBody));

  // Framing: the hostile page embeds Studio in an iframe. Observation only (no frame-ancestors / X-Frame-Options).
  await attacker.goto(`${evilOrigin}/`);
  const framed = await attacker.evaluate(async (target) => {
    const frame = document.createElement("iframe");
    frame.src = `${target}/`;
    document.body.append(frame);
    await new Promise((resolve) => frame.addEventListener("load", resolve, { once: true }));
    return "loaded";
  }, origin).catch((error) => `error:${error}`);
  const frameRequest = attackerSeen.find((s) => s.url === `${origin}/` && s.status !== undefined);
  rec.data.framing = { result: framed, frameStatus: frameRequest?.status ?? null, xFrameOptions: responses.some((r) => r.headers["x-frame-options"]), cspHasFrameAncestors: /frame-ancestors/.test(responses[0]?.headers["content-security-policy"] ?? "") };
  await attacker.close();

  // ---- CSP inside the real UI page
  await page.goto(url);
  await page.getByText("Connected to a local server.").waitFor({ timeout: 15_000 });
  evilRequests.length = 0;
  const csp = await page.evaluate(async (evilBase) => {
    const violations = [];
    document.addEventListener("securitypolicyviolation", (e) => violations.push(e.effectiveDirective));
    const out = {};
    const settle = () => new Promise((r) => setTimeout(r, 300));

    const inline = document.createElement("script");
    inline.textContent = "window.__inlineRan = true";
    document.head.append(inline);
    out.inlineScript = window.__inlineRan === true ? "RAN" : "blocked";

    try { out.setTimeoutString = (setTimeout("window.__timeoutRan = true", 0), "scheduled"); } catch { out.setTimeoutString = "blocked"; }

    const holder = document.createElement("div");
    holder.innerHTML = '<img src="data:image/png;base64,AAAA" onerror="window.__handlerRan = true">';
    document.body.append(holder);
    await settle();
    out.inlineHandler = window.__handlerRan === true ? "RAN" : "blocked";
    out.timeoutString = window.__timeoutRan === true ? "RAN" : "blocked";

    const remote = document.createElement("script");
    remote.src = `${evilBase}/remote.js`;
    document.head.append(remote);
    await settle();
    out.remoteScript = window.__remoteScriptRan === true ? "RAN" : "blocked";

    try { await fetch(`${evilBase}/data`); out.remoteFetch = "allowed"; } catch (e) { out.remoteFetch = `blocked:${e.name}`; }
    out.remoteImage = await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve("loaded");
      img.onerror = () => resolve("blocked");
      img.src = `${evilBase}/pixel.png`;
    });
    const frame = document.createElement("iframe");
    frame.src = `${evilBase}/frame`;
    document.body.append(frame);
    await settle();
    try { out.remoteFrame = frame.contentDocument?.title === "evil" ? "LOADED" : "blocked"; } catch { out.remoteFrame = "LOADED(cross-origin)"; }
    out.dataWorker = await new Promise((resolve) => {
      try {
        const worker = new Worker("data:text/javascript,postMessage(1)");
        worker.onmessage = () => resolve("RAN");
        worker.onerror = () => resolve("blocked");
        setTimeout(() => resolve("no answer"), 1500);
      } catch {
        resolve("blocked");
      }
    });
    try {
      const blobWorker = new Worker(URL.createObjectURL(new Blob(["postMessage('ok')"], { type: "text/javascript" })));
      out.blobWorker = await new Promise((resolve) => { blobWorker.onmessage = (m) => resolve(m.data); blobWorker.onerror = () => resolve("error"); });
    } catch (e) { out.blobWorker = `blocked:${e.name}`; }
    const wasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
    try { await WebAssembly.compile(wasm); out.wasmCompile = "allowed"; } catch (e) { out.wasmCompile = `blocked:${e.name}`; }
    const styled = document.createElement("div");
    styled.setAttribute("style", "color: rgb(1, 2, 3)");
    document.body.append(styled);
    out.inlineStyle = getComputedStyle(styled).color === "rgb(1, 2, 3)" ? "allowed" : "blocked";
    out.violations = [...new Set(violations)].sort();
    return out;
  }, evilOrigin);
  // eval() must be tried the way a page script would; Runtime.evaluate with the CDP exemption turned off.
  const cdp = await context.newCDPSession(page);
  const strictEval = async (expression) =>
    (await cdp.send("Runtime.evaluate", { expression: `(function(){try{return String(${expression})}catch(e){return e.name}})()`, allowUnsafeEvalBlockedByCSP: false })).result.value;
  csp.eval = (await strictEval('eval("1+1")')) === "EvalError" ? "blocked" : "RAN";
  csp.newFunction = (await strictEval('new Function("return 1")()')) === "EvalError" ? "blocked" : "RAN";
  rec.data.csp = csp;
  rec.check("CSP: inline <script> does not run", csp.inlineScript === "blocked");
  rec.check("CSP: eval() and new Function() are blocked", csp.eval === "blocked" && csp.newFunction === "blocked", `${csp.eval}/${csp.newFunction}`);
  rec.check("CSP: setTimeout(string) does not run", csp.timeoutString === "blocked");
  rec.check("CSP: inline event handler does not run", csp.inlineHandler === "blocked");
  rec.check("CSP: remote <script src> does not run", csp.remoteScript === "blocked");
  rec.check("CSP: fetch to a remote origin is blocked", csp.remoteFetch.startsWith("blocked"), csp.remoteFetch);
  rec.check("CSP: remote image is blocked", csp.remoteImage === "blocked", csp.remoteImage);
  rec.check("CSP: remote iframe is blocked", csp.remoteFrame === "blocked", csp.remoteFrame);
  rec.check("CSP: a data: worker is blocked", csp.dataWorker === "blocked", csp.dataWorker);
  rec.check("CSP: no request reached the remote origin", evilRequests.length === 0, evilRequests.join(","));
  rec.check("CSP: violations were reported for script, connect, img and frame", ["connect-src", "img-src", "script-src-elem"].every((d) => csp.violations.includes(d)), csp.violations.join(","));
  rec.check("CSP: blob: workers and wasm compilation still work (needed by ONNX Runtime Web)", csp.blobWorker === "ok" && csp.wasmCompile === "allowed", `${csp.blobWorker}/${csp.wasmCompile}`);
  const expectedNoise = ["http 401:", "Content Security Policy", "Refused to", "Failed to load resource", " csp"];
  const rest = watch.unexpected(expectedNoise);
  rec.check("no unexpected console/page errors after the blocked probes (401s and CSP blocks are the probes themselves)", rest.length === 0, rest.join(" | "));
  await context.close();
} catch (error) {
  rec.check("driver completed without an exception", false, String(error?.stack ?? error));
} finally {
  await browser.close();
  evil.close();
}
rec.finish();
