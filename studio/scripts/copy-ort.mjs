// Copies the ONNX Runtime Web runtime files (.mjs/.wasm) from node_modules to
// public/ort/ so the app serves them same-origin (no CDN; see docs/studio-engine.md).
// Runs automatically before `dev` and `build` (predev / prebuild hooks).
//
// By default only the plain wasm build is copied (~15 MB). The WebGPU build
// ("jsep", ~29 MB more) is opt-in because it is not verified in a real browser
// yet ([NV]) and the files end up inside the Python wheel:
//   ORT_COPY_WEBGPU=1 pnpm run build      (or: node scripts/copy-ort.mjs --webgpu)
//
// public/ort/ is git-ignored: never commit .wasm files (check:size forbids them).
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "node_modules", "onnxruntime-web", "dist");
const target = join(root, "public", "ort");

const withWebGpu = process.env.ORT_COPY_WEBGPU === "1" || process.argv.includes("--webgpu");
const wanted = ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];
if (withWebGpu) wanted.push("ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm");

if (!existsSync(source)) {
  console.error(`copy-ort: ${source} not found. Run "pnpm install" first.`);
  process.exit(1);
}
const missing = wanted.filter((name) => !existsSync(join(source, name)));
if (missing.length > 0) {
  console.error(`copy-ort: missing in onnxruntime-web/dist: ${missing.join(", ")} (did the package layout change?)`);
  process.exit(1);
}

mkdirSync(target, { recursive: true });
// Remove stale files (e.g. a jsep build copied earlier) so the output is exactly what was asked for.
for (const name of readdirSync(target)) {
  if (!wanted.includes(name)) rmSync(join(target, name), { force: true });
}
let copied = 0;
for (const name of wanted) {
  const from = join(source, name);
  const to = join(target, name);
  if (!existsSync(to) || statSync(to).size !== statSync(from).size) {
    copyFileSync(from, to);
    copied += 1;
  }
}
console.log(`copy-ort: ${wanted.length} files in public/ort (${copied} copied, webgpu=${withWebGpu})`);
