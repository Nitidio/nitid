/** Shared types of the browser inference engine. */

export type Task = "detect" | "segment";

/** Input contract of the ONNX (see `api-contract/onnx-meta.schema.json`). */
export interface OnnxInputMeta {
  name: "images";
  shape: [1, 3, number, number];
  dtype: "float32";
  range: "0-1";
  color: "RGB";
  resize: "squash";
}

export interface OnnxOutputMeta {
  name: string;
  dtype: string;
  format?: string;
  stride?: number;
}

/** Validated `nitid_meta` descriptor (schema_version 1). */
export interface OnnxMeta {
  schema_version: 1;
  task: Task;
  model: string;
  imgsz: number;
  input: OnnxInputMeta;
  outputs: OnnxOutputMeta[];
  /** Class names keyed by decimal class id. */
  names: Record<string, string>;
  topk: number;
  default_iou: number;
  nitid_version: string;
  sha256?: string;
}

/** `postprocessed`: labels/boxes/scores[/masks]. `raw`: pred_logits/pred_boxes[/pred_masks]. */
export type OutputLayout = "postprocessed" | "raw";

/** RGBA pixels, 8 bits per channel, row-major (what `ImageData` holds). */
export interface RgbaImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

/** A tensor as returned by ONNX Runtime: flat typed array plus dims. */
export interface TensorLike {
  data: ArrayLike<number> | ArrayLike<bigint>;
  dims: readonly number[];
}

export type NamedTensors = Record<string, TensorLike>;

/**
 * Binary instance mask, cropped to the detection box.
 *
 * Python zeroes everything outside the (clipped) box (`crop_masks_to_boxes`,
 * `dfine/utils/ops.py:11-36`), so only the box region is stored. `data` is
 * row-major, `width * height` bytes of 0/1, and its top-left pixel is
 * `(x, y)` in the coordinates of the ORIGINAL image.
 */
export interface DetectionMask {
  x: number;
  y: number;
  width: number;
  height: number;
  data: Uint8Array;
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * One detection, shaped like the `Detection` schema of `api-contract/openapi.json`
 * (`box` in pixels of the ORIGINAL image, `confidence`, `class_id`, `name`,
 * `has_mask`) plus the mask itself.
 */
export interface Detection {
  box: DetectionBox;
  confidence: number;
  class_id: number;
  name: string;
  has_mask: boolean;
  mask?: DetectionMask;
}

export interface DecodeOptions {
  /** Keep rows with `score > conf` (strict). Python default 0.5 (`dfine/model.py:121`). */
  conf?: number;
  /** Keep only these class ids; `null`/`undefined` keeps all. */
  classes?: readonly number[] | null;
  /** Mask probability threshold, `>=`. Python default 0.5 (`dfine/model.py:122`). */
  maskThreshold?: number;
  /** Upper bound for the total mask bytes of one result (default 256 MiB). */
  maxMaskBytes?: number;
}

export const DEFAULT_CONF = 0.5;
export const DEFAULT_MASK_THRESHOLD = 0.5;
export const DEFAULT_MAX_MASK_BYTES = 256 * 1024 * 1024;

/** Same limits as the REST API (`api-contract/README.md`, "Limits"). */
export const MAX_IMAGE_PIXELS = 20_000_000;
export const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
/** Safety cap for an ONNX file loaded into browser memory. */
export const MAX_MODEL_BYTES = 512 * 1024 * 1024;
