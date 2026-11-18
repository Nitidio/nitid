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
