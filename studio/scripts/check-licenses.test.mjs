// Run with: node --test scripts/check-licenses.test.mjs   (from studio/). Uses only Node's built-in test runner.
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ALLOWED_LICENSES, classify, evaluate, main, parseExpression } from "./check-licenses.mjs";

const pkg = (name, version = "1.0.0") => ({ name, versions: [version], paths: [`/x/${name}`] });

test("allowlist contains exactly the agreed permissive licences", () => {
  assert.deepEqual(
    [...ALLOWED_LICENSES].sort(),
    ["0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BlueOak-1.0.0", "ISC", "MIT"].sort(),
  );
});

test("parseExpression handles parentheses, OR and AND", () => {
  assert.deepEqual(parseExpression("(MIT OR GPL-3.0-or-later)"), [["MIT"], ["GPL-3.0-or-later"]]);
  assert.deepEqual(parseExpression("MIT AND Zlib"), [["MIT", "Zlib"]]);
  assert.deepEqual(parseExpression("  "), []);
});

test("classify: plain allowed, OR with an allowed alternative, denied cases", () => {
  assert.equal(classify("MIT"), "ok");
  assert.equal(classify("MIT OR GPL-3.0-or-later"), "ok-or");
  assert.equal(classify("GPL-3.0-only"), "denied");
  assert.equal(classify("MIT AND GPL-3.0-only"), "denied");
  assert.equal(classify("(MIT AND ISC)"), "ok");
  assert.equal(classify("Unknown"), "denied");
  assert.equal(classify("UNLICENSED"), "denied");
  assert.equal(classify("SEE LICENSE IN LICENSE.txt"), "denied");
  assert.equal(classify(""), "denied");
});

test("evaluate passes a report made only of allowed licences", () => {
  const report = { MIT: [pkg("react"), pkg("react-dom")], "Apache-2.0": [pkg("onnxruntime-web")] };
  const result = evaluate(report);

  assert.deepEqual(result.violations, []);
  assert.deepEqual(result.warnings, []);
  assert.equal(result.packages, 3);
  assert.deepEqual(result.counts, { MIT: 2, "Apache-2.0": 1 });
});

test("evaluate flags a production package outside the allowlist", () => {
  const report = { MIT: [pkg("react")], "GPL-3.0-only": [pkg("badlib", "2.1.0")] };

  assert.deepEqual(evaluate(report).violations, ["badlib@2.1.0: GPL-3.0-only"]);
});

test("evaluate accepts 'MIT OR GPL' with a warning (licensee chooses)", () => {
  const result = evaluate({ "(MIT OR GPL-3.0-or-later)": [pkg("jszip", "3.10.1")] });

  assert.deepEqual(result.violations, []);
  assert.equal(result.warnings.length, 1);
  assert.match(result.warnings[0], /jszip@3\.10\.1/);
});

test("evaluate rejects a malformed report", () => {
  assert.throws(() => evaluate({ MIT: "react" }), /unexpected report shape/);
});

function withReport(report) {
  const file = join(mkdtempSync(join(tmpdir(), "lic-")), "report.json");
  writeFileSync(file, typeof report === "string" ? report : JSON.stringify(report));
  return file;
}

function silenced(fn) {
  const original = { log: console.log, warn: console.warn, error: console.error };
  const lines = [];
  for (const key of Object.keys(original)) console[key] = (...args) => lines.push(args.join(" "));
  try {
    return { code: fn(), lines };
  } finally {
    Object.assign(console, original);
  }
}

test("main returns 0 for a clean simulated pnpm report", () => {
  const file = withReport({ MIT: [pkg("react")], ISC: [pkg("x")] });
  const { code, lines } = silenced(() => main(["--input", file]));

  assert.equal(code, 0);
  assert.match(lines.join("\n"), /check-licenses OK \(2 production packages/);
});

test("main returns 1 and names the offender", () => {
  const file = withReport({ MIT: [pkg("react")], "AGPL-3.0-only": [pkg("evil")] });
  const { code, lines } = silenced(() => main(["--input", file]));

  assert.equal(code, 1);
  assert.match(lines.join("\n"), /evil@1\.0\.0: AGPL-3\.0-only/);
});

test("main returns 2 for unreadable, non-JSON or empty input", () => {
  assert.equal(silenced(() => main(["--input", "/nonexistent/report.json"])).code, 2);
  assert.equal(silenced(() => main(["--input", withReport("not json")])).code, 2);
  assert.equal(silenced(() => main(["--input", withReport({})])).code, 2);
  assert.equal(silenced(() => main(["--input"])).code, 2);
});
