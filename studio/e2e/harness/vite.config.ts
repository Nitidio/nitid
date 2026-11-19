import { resolve } from "node:path";
import { defaultClientConditions, defineConfig } from "vite";

/**
 * Build of the parity harness (test tooling, never shipped): one page that imports the real browser
 * engine from ../../src/engine and exposes it as `window.__parity`. It uses the same ORT-web
 * conditions and worker format as the product build, and takes the wasm from `studio/public/ort/`
 * (run `node scripts/copy-ort.mjs` first). Output: $HARNESS_OUT_DIR.
 */
const studio = resolve(import.meta.dirname, "../..");

export default defineConfig({
  root: import.meta.dirname,
  base: "/",
  publicDir: resolve(studio, "public"),
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  worker: { format: "es" },
  build: {
    outDir: resolve(process.env.HARNESS_OUT_DIR ?? resolve(studio, "dist-e2e-harness")),
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
