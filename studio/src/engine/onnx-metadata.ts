/**
 * Minimal reader for `ModelProto.metadata_props` of an ONNX file.
 *
 * `onnxruntime-web` does not expose the ONNX `metadata_props`, but the nitid
 * exporter stores the descriptor there under the key `nitid_meta`
 * (`dfine/export_meta.py:META_KEY`). An ONNX file is a protobuf message, so
 * reading one top-level repeated field only needs a tiny wire-format walker:
 * big length-delimited fields (the graph, 10-100 MB) are skipped by length, not
 * parsed.
 *
 * Relevant schema (onnx.proto3): `ModelProto.metadata_props = 14`
 * (repeated `StringStringEntryProto { string key = 1; string value = 2; }`).
 */
import { EngineError } from "./errors";
import { parseMeta } from "./meta";
import type { OnnxMeta } from "./types";

export const META_KEY = "nitid_meta";
const METADATA_PROPS_FIELD = 14;

interface Cursor {
  buf: Uint8Array;
  pos: number;
}

function readVarint(cursor: Cursor): number {
  let result = 0;
  let factor = 1;
  for (let i = 0; i < 10; i++) {
    if (cursor.pos >= cursor.buf.length) throw truncated();
    const byte = cursor.buf[cursor.pos++]!;
    result += (byte & 0x7f) * factor;
    if ((byte & 0x80) === 0) return result;
    factor *= 128;
  }
  throw new EngineError("invalid_model", "ONNX file is corrupt (varint too long)");
}

function truncated(): EngineError {
  return new EngineError("invalid_model", "ONNX file is truncated or not a protobuf message");
}

function skip(cursor: Cursor, wireType: number): void {
  switch (wireType) {
    case 0:
      readVarint(cursor);
      return;
    case 1:
      cursor.pos += 8;
      break;
    case 2: {
      const length = readVarint(cursor);
      cursor.pos += length;
      break;
    }
    case 5:
      cursor.pos += 4;
      break;
    default:
      throw new EngineError("invalid_model", `ONNX file is corrupt (wire type ${wireType})`);
  }
  if (cursor.pos > cursor.buf.length) throw truncated();
}

function readEntry(buf: Uint8Array): [string, string] {
  const cursor: Cursor = { buf, pos: 0 };
  const decoder = new TextDecoder("utf-8");
  let key = "";
  let value = "";
  while (cursor.pos < buf.length) {
    const tag = readVarint(cursor);
    const field = Math.floor(tag / 8);
    const wireType = tag & 7;
    if ((field === 1 || field === 2) && wireType === 2) {
      const length = readVarint(cursor);
      if (cursor.pos + length > buf.length) throw truncated();
      const text = decoder.decode(buf.subarray(cursor.pos, cursor.pos + length));
      cursor.pos += length;
      if (field === 1) key = text;
      else value = text;
    } else {
      skip(cursor, wireType);
    }
  }
  return [key, value];
}

/** All `metadata_props` of an ONNX file as a key -> value map. */
export function readOnnxMetadataProps(bytes: Uint8Array): Record<string, string> {
  const cursor: Cursor = { buf: bytes, pos: 0 };
  const props: Record<string, string> = {};
  while (cursor.pos < bytes.length) {
    const tag = readVarint(cursor);
    const field = Math.floor(tag / 8);
    const wireType = tag & 7;
    if (field === METADATA_PROPS_FIELD && wireType === 2) {
      const length = readVarint(cursor);
      if (cursor.pos + length > bytes.length) throw truncated();
      const [key, value] = readEntry(bytes.subarray(cursor.pos, cursor.pos + length));
      props[key] = value;
      cursor.pos += length;
    } else {
      skip(cursor, wireType);
    }
  }
  return props;
}

/** The validated nitid meta embedded in an ONNX file, or `null` when absent. */
export function readEmbeddedMeta(bytes: Uint8Array): OnnxMeta | null {
  const raw = readOnnxMetadataProps(bytes)[META_KEY];
  return raw === undefined ? null : parseMeta(raw);
}

const GRAPH_FIELD = 7; // ModelProto.graph
const VALUE_INFO_FIELD = 13; // GraphProto.value_info

function varintBytes(value: number): number[] {
  const out: number[] = [];
  let v = value;
  while (v >= 128) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
}

/**
 * Returns a copy of the ONNX bytes without `graph.value_info`, or `null` when
 * the graph has none.
 *
 * Why: `value_info` is optional, purely informative shape/type metadata. Some
 * nitid exports carry a wrong entry (`_v_2148`: declared rank 0, real shape
 * [4, 256]) that Python onnxruntime 1.28 accepts with a warning but
 * onnxruntime-web 1.31.1 rejects at load time with
 * `ShapeInferenceError ... inferred=2 declared=0` (reproduced in headless
 * Chrome with the segment export). Removing the entries lets ORT infer shapes
 * itself and does not change any computation. The session retries with these
 * bytes only after a load failure of that kind.
 */
export function stripValueInfo(bytes: Uint8Array): Uint8Array | null {
  const top: Cursor = { buf: bytes, pos: 0 };
  while (top.pos < bytes.length) {
    const fieldStart = top.pos;
    const tag = readVarint(top);
    const field = Math.floor(tag / 8);
    const wireType = tag & 7;
    if (field === GRAPH_FIELD && wireType === 2) {
      const length = readVarint(top);
      const bodyStart = top.pos;
      const bodyEnd = bodyStart + length;
      if (bodyEnd > bytes.length) throw truncated();
      const graph = bytes.subarray(bodyStart, bodyEnd);
      const kept: Uint8Array[] = [];
      let removed = 0;
      const inner: Cursor = { buf: graph, pos: 0 };
      while (inner.pos < graph.length) {
        const start = inner.pos;
        const innerTag = readVarint(inner);
        const innerField = Math.floor(innerTag / 8);
        skip(inner, innerTag & 7);
        if (innerField === VALUE_INFO_FIELD) removed++;
        else kept.push(graph.subarray(start, inner.pos));
      }
      if (removed === 0) return null;
      const newLength = kept.reduce((sum, part) => sum + part.length, 0);
      const head = bytes.subarray(0, fieldStart);
      const newTag = Uint8Array.from([...varintBytes(tag), ...varintBytes(newLength)]);
      const tail = bytes.subarray(bodyEnd);
      const out = new Uint8Array(head.length + newTag.length + newLength + tail.length);
      let offset = 0;
      for (const part of [head, newTag, ...kept, tail]) {
        out.set(part, offset);
        offset += part.length;
      }
      return out;
    }
    skip(top, wireType);
  }
  return null;
}
