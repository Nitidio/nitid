/// <reference types="vite/client" />

/** Injected by vite.config.ts: "hosted" for `vite build --mode hosted`, else "auto". */
declare const __STUDIO_MODE__: "auto" | "hosted";

/**
 * Injected by vite.config.ts from `STUDIO_MODELS_BASE_URL` (build time). Empty string when not set.
 * Base URL of a folder holding `models.json` and the ONNX files for the hosted deployment.
 */
declare const __STUDIO_MODELS_BASE_URL__: string;
