/** Public surface of the browser inference engine. */
export { createInferenceClient, InferenceClient } from "./client";
export type { InitParams, WorkerLike } from "./client";
export { decodeOutputs, maskToFull } from "./decode";
export { EngineError } from "./errors";
export type { EngineErrorCode } from "./errors";
export { className, outputLayout, parseMeta, validateMeta } from "./meta";
export { readEmbeddedMeta, readOnnxMetadataProps } from "./onnx-metadata";
export { preprocessRgba } from "./preprocess";
export type { Backend, ImageInput, InitInfo, RunOptions, RunResult } from "./session";
export type { Detection, DetectionMask, OnnxMeta, RgbaImage } from "./types";
