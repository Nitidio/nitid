// Run with: node --test scripts/helpers/*.test.mjs   (from studio/; `pnpm run test:scripts`).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { headersFor, parseHeadersFile, patternToRegExp } from "./cf-headers.mjs";

const PRODUCTION = fileURLToPath(new URL("../../hosting/cloudflare/_headers", import.meta.url));

test("parses blocks, comments and detach lines in order", () => {
  const rules = parseHeadersFile("# c\n/a/*\n  X-A: 1\n  ! X-B\n\n/b\n  X-C: two: parts\n");
  assert.deepEqual(rules, [
    { pattern: "/a/*", set: [["X-A", "1"], ["X-B", null]] },
    { pattern: "/b", set: [["X-C", "two: parts"]] },
  ]);
});

test("rejects malformed files", () => {
  assert.throws(() => parseHeadersFile("  X-A: 1\n"), /before any URL pattern/);
  assert.throws(() => parseHeadersFile("/a\n  nocolon\n"), /Name: value/);
  assert.throws(() => parseHeadersFile("a/b\n  X: 1\n"), /start with/);
  assert.throws(() => parseHeadersFile("/a/*/*\n  X: 1\n"), /one splat/);
  assert.throws(() => parseHeadersFile(`/a\n  X: ${"v".repeat(2000)}\n`), /2000/);
  assert.throws(() => parseHeadersFile(Array.from({ length: 101 }, (_, i) => `/p${i}\n  X: 1\n`).join("")), /100/);
});

test("patterns: splat is greedy and may be empty, placeholders stop at a slash, dots are literal", () => {
  assert.ok(patternToRegExp("/*").test("/"));
  assert.ok(patternToRegExp("/*").test("/a/b/c.js"));
  assert.ok(patternToRegExp("/assets/*").test("/assets/x-1.js"));
  assert.ok(!patternToRegExp("/assets/*").test("/other/x.js"));
  assert.ok(patternToRegExp("/ort/*.wasm").test("/ort/a.wasm"));
  assert.ok(!patternToRegExp("/ort/*.wasm").test("/ort/a-wasm"));
  assert.ok(patternToRegExp("/:id/x").test("/7/x"));
  assert.ok(!patternToRegExp("/:id/x").test("/7/8/x"));
});

test("matching rules combine in order: join, then detach and set again", () => {
  const rules = parseHeadersFile("/*\n  X-List: a\n  Cache-Control: no-cache\n/p/*\n  X-List: b\n  ! Cache-Control\n  Cache-Control: immutable\n");
  assert.deepEqual(headersFor(rules, "/q"), { "x-list": "a", "cache-control": "no-cache" });
  assert.deepEqual(headersFor(rules, "/p/x"), { "x-list": "a, b", "cache-control": "immutable" });
});

test("the production file: security block everywhere, caching by folder", () => {
  const rules = parseHeadersFile(readFileSync(PRODUCTION, "utf8"));
  for (const path of ["/", "/index.html", "/run/browser", "/ort/ort-wasm-simd-threaded.wasm", "/models/models.json", "/assets/a-1.js"]) {
    const h = headersFor(rules, path);
    assert.equal(h["cross-origin-opener-policy"], "same-origin", path);
    assert.equal(h["cross-origin-embedder-policy"], "require-corp", path);
    assert.equal(h["cross-origin-resource-policy"], "same-origin", path);
    assert.match(h["content-security-policy"], /^default-src 'self';/, path);
  }
  assert.equal(headersFor(rules, "/assets/index-AbC123.js")["cache-control"], "public, max-age=31536000, immutable");
  assert.equal(headersFor(rules, "/index.html")["cache-control"], "no-cache");
  assert.equal(headersFor(rules, "/models/models.json")["cache-control"], "no-cache");
});
