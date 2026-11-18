// Fails if a production dependency of the Studio frontend has a licence outside the allowlist.
//
// Usage (from studio/):
//   node scripts/check-licenses.mjs              # runs `pnpm licenses list --prod --json`
//   node scripts/check-licenses.mjs --input f.json   # reads a saved report instead (tests, offline)
//   node --test scripts/check-licenses.test.mjs     # unit tests for this script
//
// What it covers
//   - Every package that `pnpm licenses list --prod` reports from the lockfile/node_modules, i.e.
//     the production dependency tree, including transitive packages and type packages.
//   - SPDX expressions: `A OR B` passes if one alternative is allowlisted (the licensee chooses;
//     a warning is printed so the choice is recorded in THIRD_PARTY_NOTICES.md); `A AND B` needs
//     every part allowlisted. Unknown, missing, UNLICENSED and "SEE LICENSE IN ..." fail.
//
// What it does NOT cover
//   - devDependencies (build/test tooling is not shipped; see docs/studio-development.md).
//   - What actually ends up in the bundle: a production dependency that is tree-shaken away still
//     counts, and a file copied in by hand does not.
//   - Whether the `license` field in a package.json is truthful, or whether a package ships its
//     own licence text. Notice obligations are tracked by hand in THIRD_PARTY_NOTICES.md.
//   - Non-npm runtime assets (ONNX Runtime Web .wasm files, model weights): see
//     docs/licensing/studio-weights.md and studio/models.lock.json.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const ALLOWED_LICENSES = Object.freeze([
  "MIT",
  "Apache-2.0",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "ISC",
  "0BSD",
  "BlueOak-1.0.0",
]);

const allowed = new Set(ALLOWED_LICENSES);

/** Strip parentheses and split an SPDX expression into OR alternatives of AND groups. */
export function parseExpression(expression) {
  const cleaned = String(expression).replace(/[()]/g, " ").trim();
  if (!cleaned) return [];
  return cleaned
    .split(/\s+OR\s+/i)
    .map((alternative) => alternative.split(/\s+AND\s+/i).map((part) => part.trim()));
}

/** Returns "ok", "ok-or" (allowed only by choosing an alternative) or "denied". */
export function classify(expression, allowlist = allowed) {
  const alternatives = parseExpression(expression);
  if (alternatives.length === 0) return "denied";
  const okAlternatives = alternatives.filter((group) => group.every((id) => allowlist.has(id)));
  if (okAlternatives.length === 0) return "denied";
  return alternatives.length > 1 ? "ok-or" : "ok";
}

/**
 * Evaluate the JSON printed by `pnpm licenses list --json`
 * (an object: licence string -> array of { name, versions, ... }).
 */
export function evaluate(report, allowlist = allowed) {
  const violations = [];
  const warnings = [];
  let packages = 0;
  const counts = {};
  for (const [license, entries] of Object.entries(report)) {
    if (!Array.isArray(entries)) throw new Error(`unexpected report shape for licence ${license}`);
    counts[license] = entries.length;
    for (const entry of entries) {
      packages += 1;
      const label = `${entry.name}@${(entry.versions ?? ["?"]).join(",")}`;
      const verdict = classify(license, allowlist);
      if (verdict === "denied") violations.push(`${label}: ${license}`);
      else if (verdict === "ok-or") warnings.push(`${label}: ${license} (alternative chosen by the licensee)`);
    }
  }
  return { violations, warnings, packages, counts };
}

function readReport(argv) {
  const index = argv.indexOf("--input");
  if (index !== -1) {
    const file = argv[index + 1];
    if (!file) throw new Error("--input needs a file");
    return JSON.parse(readFileSync(file, "utf8"));
  }
  const stdout = execFileSync("pnpm", ["licenses", "list", "--prod", "--json"], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    shell: process.platform === "win32",
  });
  return JSON.parse(stdout);
}

export function main(argv = process.argv.slice(2)) {
  let report;
  try {
    report = readReport(argv);
  } catch (error) {
    console.error(`check-licenses: cannot obtain the licence report: ${error.message}`);
    return 2;
  }
  let result;
  try {
    result = evaluate(report);
  } catch (error) {
    console.error(`check-licenses: ${error.message}`);
    return 2;
  }
  if (result.packages === 0) {
    console.error("check-licenses: the report lists no packages; refusing to pass an empty report");
    return 2;
  }
  for (const warning of result.warnings) console.warn(`warning: ${warning}`);
  if (result.violations.length > 0) {
    console.error(
      `check-licenses: ${result.violations.length} production package(s) outside the allowlist ` +
        `(${ALLOWED_LICENSES.join(", ")}):\n${result.violations.join("\n")}`,
    );
    return 1;
  }
  const summary = Object.entries(result.counts)
    .map(([license, count]) => `${license} ${count}`)
    .join(", ");
  console.log(`check-licenses OK (${result.packages} production packages: ${summary})`);
  return 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  process.exit(main());
}
