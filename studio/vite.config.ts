import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defaultClientConditions } from "vite";
import { defineConfig } from "vitest/config";

/**
 * Build modes
 *   default  -> "auto": the SPA probes GET /api/v1/health at start-up and picks
 *               local+browser or hosted (browser only). Output goes to
 *               ../nitid/studio/static so the Python package can serve it.
 *   hosted   -> `vite build --mode hosted`: no backend probing at all; the
 *               output (dist-hosted/) is a plain static site.
 * STUDIO_MODELS_BASE_URL (build time) points the browser-only mode at a folder with models.json.
 * STUDIO_OUT_DIR overrides the output directory in both modes. The directory is
 * emptied before each build, so point it at a dedicated folder.
 */
export default defineConfig(({ mode }) => {
  const hosted = mode === "hosted";
  const outDir = resolve(
    process.env.STUDIO_OUT_DIR ?? (hosted ? "dist-hosted" : "../nitid/studio/static"),
  );

  return {
    base: "/",
    plugins: [react()],
    resolve: {
      // onnxruntime-web: use its "extern wasm" entry points, which do not embed
      // the 15-29 MB .wasm files in the bundle. The runtime loads them from
      // /ort/ (copied by scripts/copy-ort.mjs) through env.wasm.wasmPaths.
      conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
    },
    define: {
      __STUDIO_MODE__: JSON.stringify(hosted ? "hosted" : "auto"),
      // Base URL of the hosted model catalogue ("" = none configured). Never hardcode a vendor URL here.
      __STUDIO_MODELS_BASE_URL__: JSON.stringify(process.env.STUDIO_MODELS_BASE_URL ?? ""),
    },
    server: {
      port: 5173,
      proxy: {
        "/api": { target: "http://127.0.0.1:8000", changeOrigin: false },
      },
    },
    // The inference worker (src/workers/inference.worker.ts) uses dynamic import()
    // to load onnxruntime-web, which needs ES-module workers (the default "iife"
    // format cannot code-split). Same-origin only: no CDN.
    worker: { format: "es" },
    build: {
      outDir,
      emptyOutDir: true,
      sourcemap: false,
      // Never inline assets as data: URIs (keeps CSP simple and files cacheable).
      assetsInlineLimit: 0,
      rollupOptions: {
        output: {
          entryFileNames: "assets/[name]-[hash].js",
          chunkFileNames: "assets/[name]-[hash].js",
          assetFileNames: "assets/[name]-[hash][extname]",
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      restoreMocks: true,
    },
  };
});
