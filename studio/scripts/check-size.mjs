// Fails if a git-tracked file under studio/ is a binary/model artefact or exceeds 1 MB.
// Usage: pnpm run check:size   (run from studio/; needs git)
import { execFileSync } from "node:child_process";
import { statSync } from "node:fs";
import { basename } from "node:path";

const LIMIT = 1024 * 1024;
const ALLOWED_OVER_LIMIT = new Set(["pnpm-lock.yaml"]);
const FORBIDDEN_EXT = /\.(wasm|onnx|pth|pt|engine|torchscript|bin|zip|gz|tgz)$/i;

const files = execFileSync("git", ["ls-files", "-z", "--", "."], { encoding: "utf8" })
  .split("\0")
  .filter(Boolean);

const problems = [];
for (const file of files) {
  if (FORBIDDEN_EXT.test(file)) problems.push(`${file}: forbidden file type`);
  const size = statSync(file).size;
  if (size > LIMIT && !ALLOWED_OVER_LIMIT.has(basename(file))) {
    problems.push(`${file}: ${size} bytes exceeds 1 MB`);
  }
}

if (problems.length > 0) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`check:size OK (${files.length} tracked files under studio/)`);
