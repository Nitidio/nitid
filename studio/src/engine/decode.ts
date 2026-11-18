/**
 * Decoding of the ONNX outputs into detections, replicating Python.
 *
 * Reference: `DFINEPredictor._postprocess` (`dfine/predictor.py:377-470`) fed by
 * `DFINEPostProcessor` (`dfine/nn/postprocessor.py`). What the Python code does
 * in the default path (`augment=False`), and what this file therefore does:
 *
 * 1. Rows with `scores > conf` are kept (STRICT `>`; `predictor.py:419`). Default
 *    `conf = 0.5` (`dfine/model.py:121`), not 0.25.
 * 2. If `classes` is given, rows whose label is in it are kept (`predictor.py:421-431`).
 * 3. NO NMS is applied. The only suppression in Python is the cross-view merge of
 *    `augment=True` (flip TTA, `predictor.py:433-458`), which needs two forward
 *    passes and is not implemented here (`[NV]` v1 scope). `iou` is therefore
 *    unused (the meta's `default_iou` is informational). Overlapping boxes of
 *    one class are all kept, exactly like Python; the top-K selection already
 *    happened in the graph (`postprocessor.py:53,68-69`, `topk` of the meta).
 * 4. Boxes: the graph returns `xyxy` in pixels of the model INPUT (S x S, the
 *    deploy wrapper passes `(W, H) = input size`, `exporter.py` `DeployModel`).
 *    They are rescaled to the original image with `x * W / S`, `y * H / S`
 *    (Python multiplies the normalised box by `(W, H)` directly,
 *    `postprocessor.py:47-48`; the two are equal up to float rounding), then
 *    clipped to `[0, W] x [0, H]` (`clip_boxes`, `dfine/utils/ops.py:59-64`).
 *    Filtering happens BEFORE clipping, like Python.
 * 5. Masks (segment): the graph returns SIGMOID PROBABILITIES at `S/4`
 *    (`decoder.py:1198-1199`, eval mode), already gathered per selected query.
 *    Python resizes them to the original `(H, W)` with
 *    `F.interpolate(mode="bilinear", align_corners=False)` (no antialiasing),
 *    thresholds with `>= mask_threshold` (default 0.5, `model.py:122`) and then
 *    zeroes everything outside the clipped box (`crop_masks_to_boxes`,
 *    `ops.py:11-36`: pixel `x` is inside when `x1 <= x < x2`). Only the box
 *    region is computed and stored (`DetectionMask`), which is identical to the
 *    full-size Python mask but needs far less memory.
 *
 * The raw layout (`postprocess=False` exports) is also supported by replaying
 * the post-processor: sigmoid, top-K over the flattened `[queries * classes]`
 * scores, `label = idx % C`, `query = idx // C` (`postprocessor.py:55-62`).
 * `[NV]`: it assumes `use_focal_loss=True` (the nitid default); the meta does
 * not record that flag, and the softmax variant (`postprocessor.py:63-77`) is
 * not implemented.
 */
import { EngineError } from "./errors";
import { className, outputLayout, outputNames } from "./meta";
import {
  DEFAULT_CONF,
  DEFAULT_MASK_THRESHOLD,
  DEFAULT_MAX_MASK_BYTES,
  type DecodeOptions,
  type Detection,
  type DetectionMask,
  type NamedTensors,
  type OnnxMeta,
  type TensorLike,
} from "./types";

interface Candidate {
  /** Row in the (post-processed) output, or query index for the raw layout. */
  maskRow: number;
  label: number;
  score: number;
  /** Box in pixels of the ORIGINAL image, not yet clipped. */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface MaskSource {
  data: ArrayLike<number>;
  rows: number;
  height: number;
  width: number;
}

function requireTensor(outputs: NamedTensors, name: string): TensorLike {
  const tensor = outputs[name];
  if (!tensor) throw new EngineError("unsupported_output", `Model output "${name}" is missing`);
  return tensor;
}

function numberAt(data: ArrayLike<number> | ArrayLike<bigint>, index: number): number {
  return Number(data[index]);
}

function expectDims(name: string, tensor: TensorLike, rank: number, tail: readonly (number | null)[]): void {
  const dims = tensor.dims;
  const ok =
    dims.length === rank &&
    dims[0] === 1 &&
    tail.every((expected, i) => expected === null || dims[i + 1] === expected);
  if (!ok) {
    throw new EngineError(
      "unsupported_output",
      `Output "${name}" has dims [${dims.join(", ")}]; expected batch 1 and rank ${rank}`,
    );
  }
}

/**
 * Decodes the outputs of one inference into detections.
 *
 * @param original size of the image that was fed to preprocessing (before the squash resize).
 */
export function decodeOutputs(
  meta: OnnxMeta,
  outputs: NamedTensors,
  original: { width: number; height: number },
  options: DecodeOptions = {},
): Detection[] {
  const conf = options.conf ?? DEFAULT_CONF;
  const maskThreshold = options.maskThreshold ?? DEFAULT_MASK_THRESHOLD;
  const maxMaskBytes = options.maxMaskBytes ?? DEFAULT_MAX_MASK_BYTES;
  if (!Number.isFinite(conf) || conf < 0 || conf > 1) {
    throw new EngineError("invalid_options", `conf must be in [0, 1], got ${conf}`);
  }
  if (!Number.isFinite(maskThreshold) || maskThreshold < 0 || maskThreshold > 1) {
    throw new EngineError("invalid_options", `maskThreshold must be in [0, 1], got ${maskThreshold}`);
  }
  const classSet = options.classes == null ? null : new Set(options.classes);

  const { width, height } = original;
  const hasMasks = meta.task === "segment";
  let candidates: Candidate[];
  let masks: MaskSource | null = null;

  if (outputLayout(meta) === "postprocessed") {
    ({ candidates, masks } = readPostprocessed(meta, outputs, width, height, hasMasks));
  } else {
    ({ candidates, masks } = readRaw(meta, outputs, width, height, hasMasks));
  }

  const kept = candidates.filter(
    (c) => c.score > conf && (classSet === null || classSet.has(c.label)),
  );

  // Clip AFTER filtering (predictor.py:460 `clip_boxes` runs after the filters).
  const clipped = kept.map((c) => ({
    ...c,
    x1: clamp(c.x1, 0, width),
    y1: clamp(c.y1, 0, height),
    x2: clamp(c.x2, 0, width),
    y2: clamp(c.y2, 0, height),
  }));

  // Memory guard: total bytes of all masks of this result.
  if (masks) {
    let bytes = 0;
    for (const c of clipped) {
      const rect = maskRect(c, width, height);
      bytes += rect.w * rect.h;
    }
    if (bytes > maxMaskBytes) {
      throw new EngineError(
        "mask_budget_exceeded",
        `Masks of ${clipped.length} detections need ${(bytes / 2 ** 20).toFixed(0)} MiB (limit ${(maxMaskBytes / 2 ** 20).toFixed(0)} MiB). ` +
          "Raise the confidence threshold or use a smaller image.",
      );
    }
  }

  return clipped.map((c): Detection => {
    const detection: Detection = {
      box: { x1: c.x1, y1: c.y1, x2: c.x2, y2: c.y2 },
      confidence: c.score,
      class_id: c.label,
      name: className(meta, c.label),
      has_mask: masks !== null,
    };
    if (masks) detection.mask = buildMask(masks, c, width, height, maskThreshold);
    return detection;
  });
}

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

function readPostprocessed(
  meta: OnnxMeta,
  outputs: NamedTensors,
  width: number,
  height: number,
  hasMasks: boolean,
): { candidates: Candidate[]; masks: MaskSource | null } {
  const [labelsName, boxesName, scoresName] = outputNames(meta) as [string, string, string];
  const labels = requireTensor(outputs, labelsName);
  const boxes = requireTensor(outputs, boxesName);
  const scores = requireTensor(outputs, scoresName);
  const k = labels.dims[1] ?? 0;
  expectDims(labelsName, labels, 2, [null]);
  expectDims(boxesName, boxes, 3, [k, 4]);
  expectDims(scoresName, scores, 2, [k]);

  const size = meta.imgsz;
  const sx = width / size;
  const sy = height / size;
  const candidates: Candidate[] = [];
  for (let i = 0; i < k; i++) {
    candidates.push({
      maskRow: i,
      label: numberAt(labels.data, i),
      score: numberAt(scores.data, i),
      x1: numberAt(boxes.data, i * 4) * sx,
      y1: numberAt(boxes.data, i * 4 + 1) * sy,
      x2: numberAt(boxes.data, i * 4 + 2) * sx,
      y2: numberAt(boxes.data, i * 4 + 3) * sy,
    });
  }

  let masks: MaskSource | null = null;
  if (hasMasks) {
    const tensor = requireTensor(outputs, "masks");
    expectDims("masks", tensor, 4, [k, null, null]);
    masks = {
      data: tensor.data as ArrayLike<number>,
      rows: k,
      height: tensor.dims[2]!,
      width: tensor.dims[3]!,
    };
  }
  return { candidates, masks };
}

function readRaw(
  meta: OnnxMeta,
  outputs: NamedTensors,
  width: number,
  height: number,
  hasMasks: boolean,
): { candidates: Candidate[]; masks: MaskSource | null } {
  const logits = requireTensor(outputs, "pred_logits");
  const boxes = requireTensor(outputs, "pred_boxes");
  expectDims("pred_logits", logits, 3, [null, null]);
  const queries = logits.dims[1]!;
  const classes = logits.dims[2]!;
  expectDims("pred_boxes", boxes, 3, [queries, 4]);

  // sigmoid + top-K over the flattened [queries * classes] (postprocessor.py:55-58).
  const total = queries * classes;
  const probabilities = new Float64Array(total);
  for (let i = 0; i < total; i++) probabilities[i] = 1 / (1 + Math.exp(-numberAt(logits.data, i)));
  const order = Array.from({ length: total }, (_, i) => i).sort(
    (a, b) => probabilities[b]! - probabilities[a]! || a - b,
  );
  const topk = Math.min(meta.topk, total);

  const candidates: Candidate[] = [];
  for (const index of order.slice(0, topk)) {
    const query = Math.floor(index / classes);
    const cx = numberAt(boxes.data, query * 4);
    const cy = numberAt(boxes.data, query * 4 + 1);
    const w = numberAt(boxes.data, query * 4 + 2);
    const h = numberAt(boxes.data, query * 4 + 3);
    candidates.push({
      maskRow: query,
      label: index % classes,
      score: probabilities[index]!,
      // cxcywh (normalised) -> xyxy -> times (W, H): postprocessor.py:46-48.
      x1: (cx - w / 2) * width,
      y1: (cy - h / 2) * height,
      x2: (cx + w / 2) * width,
      y2: (cy + h / 2) * height,
    });
  }

  let masks: MaskSource | null = null;
  if (hasMasks) {
    const tensor = requireTensor(outputs, "pred_masks");
    expectDims("pred_masks", tensor, 4, [queries, null, null]);
    masks = {
      data: tensor.data as ArrayLike<number>,
      rows: queries,
      height: tensor.dims[2]!,
      width: tensor.dims[3]!,
    };
  }
  return { candidates, masks };
}

/** Integer pixel rectangle `[x0, x1) x [y0, y1)` inside the clipped box (crop_masks_to_boxes). */
function maskRect(
  c: { x1: number; y1: number; x2: number; y2: number },
  width: number,
  height: number,
): { x: number; y: number; w: number; h: number } {
  // Pixel x is inside when x1 <= x < x2, i.e. x in [ceil(x1), ceil(x2) - 1].
  const x0 = Math.max(0, Math.ceil(c.x1));
  const x1 = Math.min(width, Math.ceil(c.x2));
  const y0 = Math.max(0, Math.ceil(c.y1));
  const y1 = Math.min(height, Math.ceil(c.y2));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

/** Source index and weight of `F.interpolate(mode="bilinear", align_corners=False)`. */
function axisTaps(outSize: number, inSize: number, start: number, count: number) {
  const idx0 = new Int32Array(count);
  const idx1 = new Int32Array(count);
  const lambda = new Float64Array(count);
  const scale = inSize / outSize;
  for (let i = 0; i < count; i++) {
    let source = scale * (start + i + 0.5) - 0.5;
    if (source < 0) source = 0;
    const i0 = Math.min(Math.floor(source), inSize - 1);
    idx0[i] = i0;
    idx1[i] = i0 + (i0 < inSize - 1 ? 1 : 0);
    lambda[i] = clamp(source - i0, 0, 1);
  }
  return { idx0, idx1, lambda };
}

function buildMask(
  source: MaskSource,
  c: Candidate & { x1: number; y1: number; x2: number; y2: number },
  width: number,
  height: number,
  threshold: number,
): DetectionMask {
  const rect = maskRect(c, width, height);
  const data = new Uint8Array(rect.w * rect.h);
  if (rect.w === 0 || rect.h === 0) return { x: rect.x, y: rect.y, width: rect.w, height: rect.h, data };

  const base = c.maskRow * source.height * source.width;
  const xs = axisTaps(width, source.width, rect.x, rect.w);
  const ys = axisTaps(height, source.height, rect.y, rect.h);
  const at = (row: number, col: number) => Number(source.data[base + row * source.width + col]);
  for (let j = 0; j < rect.h; j++) {
    const r0 = ys.idx0[j]!;
    const r1 = ys.idx1[j]!;
    const ly = ys.lambda[j]!;
    for (let i = 0; i < rect.w; i++) {
      const c0 = xs.idx0[i]!;
      const c1 = xs.idx1[i]!;
      const lx = xs.lambda[i]!;
      const top = at(r0, c0) * (1 - lx) + at(r0, c1) * lx;
      const bottom = at(r1, c0) * (1 - lx) + at(r1, c1) * lx;
      data[j * rect.w + i] = top * (1 - ly) + bottom * ly >= threshold ? 1 : 0;
    }
  }
  return { x: rect.x, y: rect.y, width: rect.w, height: rect.h, data };
}

/** Expands a box-cropped mask to a full `width * height` 0/1 array (tests, export). */
export function maskToFull(mask: DetectionMask, width: number, height: number): Uint8Array {
  const full = new Uint8Array(width * height);
  for (let j = 0; j < mask.height; j++) {
    full.set(mask.data.subarray(j * mask.width, (j + 1) * mask.width), (mask.y + j) * width + mask.x);
  }
  return full;
}
