/**
 * Parsing and validation of the `nitid_meta` descriptor.
 *
 * The rules mirror `api-contract/onnx-meta.schema.json` (JSON Schema 2020-12)
 * plus the cross-field rule the schema cannot express (`input.shape` equals
 * `[1, 3, imgsz, imgsz]`). The validator is written by hand on purpose and
 * `ajv` is NOT used:
 *
 * - `ajv` compiles schemas with `new Function`, which needs `'unsafe-eval'` in
 *   the Content-Security-Policy; the Studio CSP is
 *   `script-src 'self' 'wasm-unsafe-eval'` and must stay that way.
 * - the schema is small and frozen (v1), so ~150 lines are cheaper than a
 *   dependency in the worker bundle.
 *
 * `meta.test.ts` runs this validator against every `meta.*.json` fixture of the
 * contract, so drift from the schema is caught by the tests.
 */
import { EngineError } from "./errors";
import type { OnnxMeta, OnnxOutputMeta, OutputLayout } from "./types";

const POSTPROCESSED: Record<string, readonly string[]> = {
  detect: ["labels", "boxes", "scores"],
  segment: ["labels", "boxes", "scores", "masks"],
};
const RAW: Record<string, readonly string[]> = {
  detect: ["pred_logits", "pred_boxes"],
  segment: ["pred_logits", "pred_boxes", "pred_masks"],
};

const TOP_KEYS = new Set([
  "schema_version",
  "task",
  "model",
  "imgsz",
  "input",
  "outputs",
  "names",
  "topk",
  "default_iou",
  "nitid_version",
  "sha256",
]);
const INPUT_KEYS = ["name", "shape", "dtype", "range", "color", "resize"] as const;
const INPUT_CONSTS: Record<(typeof INPUT_KEYS)[number], unknown> = {
  name: "images",
  shape: undefined,
  dtype: "float32",
  range: "0-1",
  color: "RGB",
  resize: "squash",
};

/** Output descriptor: allowed keys and constant values, by output name. */
const OUTPUT_SPEC: Record<string, { dtype: string; format?: string; stride?: number }> = {
  labels: { dtype: "int64" },
  boxes: { dtype: "float32", format: "xyxy_input_px" },
  scores: { dtype: "float32" },
  masks: { dtype: "float32", stride: 4 },
  pred_logits: { dtype: "float32" },
  pred_boxes: { dtype: "float32", format: "cxcywh_norm" },
  pred_masks: { dtype: "float32", stride: 4 },
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

/** Returns the list of problems (empty when the meta is valid). */
export function validateMeta(input: unknown): string[] {
  const problems: string[] = [];
  if (!isObject(input)) return ["meta must be a JSON object"];

  for (const key of Object.keys(input)) {
    if (!TOP_KEYS.has(key)) problems.push(`unexpected property "${key}"`);
  }
  for (const key of [
    "schema_version",
    "task",
    "model",
    "imgsz",
    "input",
    "outputs",
    "names",
    "topk",
    "default_iou",
    "nitid_version",
  ]) {
    if (!(key in input)) problems.push(`missing property "${key}"`);
  }

  if ("schema_version" in input && input.schema_version !== 1) {
    problems.push(`schema_version must be 1, got ${JSON.stringify(input.schema_version)}`);
  }
  const task = input.task;
  if ("task" in input && task !== "detect" && task !== "segment") {
    problems.push(`task must be "detect" or "segment", got ${JSON.stringify(task)}`);
  }
  if ("model" in input && (typeof input.model !== "string" || input.model.length === 0)) {
    problems.push("model must be a non-empty string");
  }
  const imgsz = input.imgsz;
  if ("imgsz" in input) {
    if (!isInt(imgsz) || imgsz < 32 || imgsz > 4096 || imgsz % 32 !== 0) {
      problems.push("imgsz must be an integer in [32, 4096] and a multiple of 32");
    }
  }

  if ("input" in input) validateInput(input.input, imgsz, problems);
  if ("outputs" in input) validateOutputs(input.outputs, task, problems);

  if ("names" in input) {
    const names = input.names;
    if (!isObject(names) || Object.keys(names).length === 0) {
      problems.push("names must be a non-empty object");
    } else {
      for (const [key, value] of Object.entries(names)) {
        if (!/^(0|[1-9][0-9]*)$/.test(key)) problems.push(`names key "${key}" is not a decimal class id`);
        if (typeof value !== "string") problems.push(`names["${key}"] must be a string`);
      }
    }
  }
  if ("topk" in input && (!isInt(input.topk) || input.topk < 1)) {
    problems.push("topk must be an integer >= 1");
  }
  if (
    "default_iou" in input &&
    (typeof input.default_iou !== "number" || input.default_iou < 0 || input.default_iou > 1)
  ) {
    problems.push("default_iou must be a number in [0, 1]");
  }
  if (
    "nitid_version" in input &&
    (typeof input.nitid_version !== "string" || !/^[0-9]+\.[0-9]+\.[0-9]+/.test(input.nitid_version))
  ) {
    problems.push("nitid_version must look like MAJOR.MINOR.PATCH");
  }
  if ("sha256" in input && (typeof input.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(input.sha256))) {
    problems.push("sha256 must be 64 lowercase hex characters");
  }
  return problems;
}

function validateInput(value: unknown, imgsz: unknown, problems: string[]): void {
  if (!isObject(value)) {
    problems.push("input must be an object");
    return;
  }
  for (const key of Object.keys(value)) {
    if (!(INPUT_KEYS as readonly string[]).includes(key)) problems.push(`unexpected input property "${key}"`);
  }
  for (const key of INPUT_KEYS) {
    if (!(key in value)) {
      problems.push(`missing input property "${key}"`);
    } else if (key !== "shape" && value[key] !== INPUT_CONSTS[key]) {
      problems.push(`input.${key} must be ${JSON.stringify(INPUT_CONSTS[key])}, got ${JSON.stringify(value[key])}`);
    }
  }
  const shape = value.shape;
  if ("shape" in value) {
    if (
      !Array.isArray(shape) ||
      shape.length !== 4 ||
      shape[0] !== 1 ||
      shape[1] !== 3 ||
      !isInt(shape[2]) ||
      !isInt(shape[3]) ||
      shape[2] < 32 ||
      shape[3] < 32
    ) {
      problems.push("input.shape must be the static shape [1, 3, S, S] with S >= 32");
    } else if (isInt(imgsz) && (shape[2] !== imgsz || shape[3] !== imgsz)) {
      // Cross-field rule that JSON Schema cannot express (api-contract README).
      problems.push(`input.shape ${JSON.stringify(shape)} does not match imgsz ${imgsz}`);
    }
  }
}

function validateOutputs(value: unknown, task: unknown, problems: string[]): void {
  if (!Array.isArray(value)) {
    problems.push("outputs must be an array");
    return;
  }
  const names: string[] = [];
  for (const [index, output] of value.entries()) {
    if (!isObject(output) || typeof output.name !== "string" || !(output.name in OUTPUT_SPEC)) {
      problems.push(`outputs[${index}] is not a known output descriptor`);
      continue;
    }
    names.push(output.name);
    const spec = OUTPUT_SPEC[output.name]!;
    const allowed = new Set(["name", "dtype", ...(spec.format ? ["format"] : []), ...(spec.stride ? ["stride"] : [])]);
    for (const key of Object.keys(output)) {
      if (!allowed.has(key)) problems.push(`outputs[${index}] has unexpected property "${key}"`);
    }
    if (output.dtype !== spec.dtype) problems.push(`outputs[${index}].dtype must be "${spec.dtype}"`);
    if (spec.format && output.format !== spec.format) problems.push(`outputs[${index}].format must be "${spec.format}"`);
    if (spec.stride && output.stride !== spec.stride) problems.push(`outputs[${index}].stride must be ${spec.stride}`);
  }
  if (task === "detect" || task === "segment") {
    const post = POSTPROCESSED[task]!;
    const raw = RAW[task]!;
    const same = (expected: readonly string[]) =>
      names.length === expected.length && expected.every((name, i) => names[i] === name);
    if (!same(post) && !same(raw)) {
      problems.push(
        `outputs for task "${task}" must be exactly [${post.join(", ")}] or [${raw.join(", ")}], got [${names.join(", ")}]`,
      );
    }
  }
}

/** Validates and returns the meta, or throws `EngineError("invalid_meta")`. */
export function parseMeta(input: unknown): OnnxMeta {
  let value = input;
  if (typeof input === "string") {
    try {
      value = JSON.parse(input);
    } catch (error) {
      throw new EngineError("invalid_meta", "nitid_meta is not valid JSON", { cause: error });
    }
  }
  const problems = validateMeta(value);
  if (problems.length > 0) {
    throw new EngineError("invalid_meta", `Invalid nitid_meta: ${problems.join("; ")}`);
  }
  return value as OnnxMeta;
}

/** Which of the two output contracts the meta describes. */
export function outputLayout(meta: OnnxMeta): OutputLayout {
  return meta.outputs[0]?.name === "labels" ? "postprocessed" : "raw";
}

export function outputNames(meta: OnnxMeta): string[] {
  return meta.outputs.map((output: OnnxOutputMeta) => output.name);
}

/** Class name of an id; falls back to the decimal id when the map lacks it. */
export function className(meta: OnnxMeta, classId: number): string {
  return meta.names[String(classId)] ?? String(classId);
}
