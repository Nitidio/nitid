// Test fixtures shared by the browser smoke tests (smoke-unified.mjs, smoke-hosted.mjs): a white PNG and
// a tiny ONNX model written as protobuf by hand. No `onnx` package, no network, no committed binaries.
import { createHash } from "node:crypto";
import { deflateSync } from "node:zlib";

/* ---------------- a 200x100 white PNG ---------------- */
function crc32(buf) {
  let c, crc = ~0;
  for (const b of buf) {
    c = (crc ^ b) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return ~crc >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
export function whitePng(w, h) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const row = Buffer.concat([Buffer.from([0]), Buffer.alloc(w * 3, 255)]);
  const raw = Buffer.concat(Array.from({ length: h }, () => row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}
export const PNG = whitePng(200, 100);

/* ---------------- a tiny ONNX file, written as protobuf by hand ----------------
 * Constant outputs in the post-processed layout (labels, boxes, scores) for a 64x64 input, so the
 * result is known exactly and a real ONNX Runtime Web session runs it. No `onnx` package needed. */
export const S = 64;
const varint = (value) => {
  let n = BigInt(value);
  const out = [];
  do {
    let b = Number(n & 0x7fn);
    n >>= 7n;
    if (n) b |= 0x80;
    out.push(b);
  } while (n);
  return Buffer.from(out);
};
const tag = (field, wire) => varint((field << 3) | wire);
const lenField = (field, buf) => Buffer.concat([tag(field, 2), varint(buf.length), buf]);
const strField = (field, s) => lenField(field, Buffer.from(s, "utf8"));
const intField = (field, n) => Buffer.concat([tag(field, 0), varint(n)]);
const tensor = (name, dims, dataType, raw) =>
  Buffer.concat([...dims.map((d) => intField(1, d)), intField(2, dataType), strField(8, name), lenField(9, raw)]);
const valueInfo = (name, elemType, dims) =>
  Buffer.concat([
    strField(1, name),
    lenField(2, lenField(1, Buffer.concat([intField(1, elemType), lenField(2, Buffer.concat(dims.map((d) => lenField(1, intField(1, d)))))]))),
  ]);
const f32 = (xs) => Buffer.from(new Float32Array(xs).buffer);
const i64 = (xs) => Buffer.from(new BigInt64Array(xs.map(BigInt)).buffer);

export const NAMES = { 0: "person", 1: "bicycle", 2: "car" };
export const BOXES_IN = [8, 8, 24, 40, 32, 12, 60, 52]; // xyxy in the 64x64 model input
export const metaFor = (sha256) => ({
  schema_version: 1, task: "detect", model: "nitid1s", imgsz: S,
  input: { name: "images", shape: [1, 3, S, S], dtype: "float32", range: "0-1", color: "RGB", resize: "squash" },
  outputs: [{ name: "labels", dtype: "int64" }, { name: "boxes", dtype: "float32", format: "xyxy_input_px" }, { name: "scores", dtype: "float32" }],
  names: NAMES, topk: 2, default_iou: 0.85, nitid_version: "0.0.0", ...(sha256 ? { sha256 } : {}),
});
export function buildOnnx() {
  const node = (inp, out) => Buffer.concat([strField(1, inp), strField(2, out), strField(4, "Identity")]);
  const graph = Buffer.concat([
    lenField(1, node("c_labels", "labels")),
    lenField(1, node("c_boxes", "boxes")),
    lenField(1, node("c_scores", "scores")),
    strField(2, "tiny"),
    lenField(5, tensor("c_labels", [1, 2], 7, i64([0, 2]))),
    lenField(5, tensor("c_boxes", [1, 2, 4], 1, f32(BOXES_IN))),
    lenField(5, tensor("c_scores", [1, 2], 1, f32([0.9, 0.6]))),
    lenField(11, valueInfo("images", 1, [1, 3, S, S])),
    lenField(12, valueInfo("labels", 7, [1, 2])),
    lenField(12, valueInfo("boxes", 1, [1, 2, 4])),
    lenField(12, valueInfo("scores", 1, [1, 2])),
  ]);
  const metaEntry = Buffer.concat([strField(1, "nitid_meta"), strField(2, JSON.stringify(metaFor()))]);
  return Buffer.concat([
    intField(1, 8),
    strField(2, "smoke"),
    lenField(7, graph),
    lenField(8, Buffer.concat([strField(1, ""), intField(2, 17)])),
    lenField(14, metaEntry),
  ]);
}
export const ONNX = buildOnnx();
export const SHA = createHash("sha256").update(ONNX).digest("hex");

