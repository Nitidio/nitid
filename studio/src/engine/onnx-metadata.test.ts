import { describe, expect, it } from "vitest";
import { EngineError } from "./errors";
import { META_KEY, readEmbeddedMeta, readOnnxMetadataProps } from "./onnx-metadata";
import { loadFixture } from "./test-utils";

// Tiny protobuf writer, enough to build a ModelProto-shaped buffer by hand.
const enc = new TextEncoder();
function varint(value: number): number[] {
  const out: number[] = [];
  let v = value;
  while (v >= 128) {
    out.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
}
const lenField = (field: number, bytes: Uint8Array | number[]) => [...varint(field * 8 + 2), ...varint(bytes.length), ...bytes];
const str = (field: number, text: string) => lenField(field, enc.encode(text));
const entry = (key: string, value: string) => lenField(14, [...str(1, key), ...str(2, value)]);

function fakeModel(parts: number[][]): Uint8Array {
  return Uint8Array.from(parts.flat());
}

describe("onnx-metadata: reading metadata_props without parsing the graph", () => {
  const meta = JSON.stringify(loadFixture("detect").meta);

  it("finds nitid_meta among other fields, skipping a large graph by length", () => {
    const graph = lenField(7, new Uint8Array(100_000).fill(0xff)); // not valid protobuf: must be skipped, not parsed
    const bytes = fakeModel([
      [0x08, 0x08], // ir_version = 8 (varint field 1)
      str(2, "pytorch"), // producer_name
      graph,
      entry("other", "value"),
      entry(META_KEY, meta),
    ]);
    const props = readOnnxMetadataProps(bytes);
    expect(props.other).toBe("value");
    expect(props[META_KEY]).toBe(meta);
    expect(readEmbeddedMeta(bytes)?.task).toBe("detect");
  });

  it("handles non-ASCII text and multi-byte lengths", () => {
    const long = "ñ".repeat(200); // 400 bytes: two-byte varint length
    const props = readOnnxMetadataProps(fakeModel([entry("k", long)]));
    expect(props.k).toBe(long);
  });

  it("returns null when the file has no nitid_meta", () => {
    expect(readEmbeddedMeta(fakeModel([entry("other", "x")]))).toBeNull();
  });

  it("throws invalid_meta when nitid_meta is not a valid meta", () => {
    const bytes = fakeModel([entry(META_KEY, '{"schema_version":1}')]);
    expect(() => readEmbeddedMeta(bytes)).toThrowError(/Invalid nitid_meta/);
  });

  it("reports a truncated file as invalid_model", () => {
    const bytes = fakeModel([entry(META_KEY, meta)]).slice(0, 40);
    try {
      readOnnxMetadataProps(bytes);
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(EngineError);
      expect((error as EngineError).code).toBe("invalid_model");
    }
  });

  it("reports an unknown wire type as invalid_model", () => {
    expect(() => readOnnxMetadataProps(Uint8Array.from([0x0b]))).toThrowError(/wire type/); // field 1, wire type 3
  });

  it("an empty file has no metadata", () => {
    expect(readOnnxMetadataProps(new Uint8Array())).toEqual({});
  });
});
