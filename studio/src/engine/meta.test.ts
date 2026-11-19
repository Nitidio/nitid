import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { EngineError } from "./errors";
import { className, outputLayout, outputNames, parseMeta, validateMeta } from "./meta";
import { loadFixture } from "./test-utils";

const FIXTURES = resolve(import.meta.dirname, "../../api-contract/fixtures");
const metaFiles = readdirSync(FIXTURES).filter((name) => /^meta\..*\.json$/.test(name));
const read = (name: string) => JSON.parse(readFileSync(resolve(FIXTURES, name), "utf8")) as unknown;

describe("meta: the validator agrees with the API contract fixtures", () => {
  it("found the contract fixtures", () => {
    expect(metaFiles.length).toBeGreaterThan(30);
  });

  for (const name of metaFiles) {
    if (name.endsWith(".valid.json")) {
      it(`accepts ${name}`, () => {
        expect(validateMeta(read(name))).toEqual([]);
      });
    } else {
      // `meta.invalid.*` fail the schema; `meta.invalid-semantic.*` fail the cross-field rule.
      it(`rejects ${name}`, () => {
        expect(validateMeta(read(name)).length).toBeGreaterThan(0);
      });
    }
  }
});

describe("meta: parseMeta", () => {
  const valid = read("meta.detect.valid.json") as Record<string, unknown>;

  it("returns the validated object", () => {
    const meta = parseMeta(valid);
    expect(meta.task).toBe("detect");
    expect(meta.imgsz).toBe(640);
  });

  it("accepts a JSON string (as stored in ONNX metadata_props)", () => {
    expect(parseMeta(JSON.stringify(valid)).model).toBe(valid.model);
  });

  it("rejects malformed JSON text with invalid_meta", () => {
    expect(() => parseMeta("{nope")).toThrowError(EngineError);
    try {
      parseMeta("{nope");
    } catch (error) {
      expect((error as EngineError).code).toBe("invalid_meta");
    }
  });

  it("lists every problem in the message", () => {
    const broken = { ...valid, task: "pose", topk: 0 };
    expect(() => parseMeta(broken)).toThrowError(/task must be.*topk must be/s);
  });

  it("rejects non-objects", () => {
    for (const value of [null, 3, "x", [], undefined]) {
      expect(validateMeta(value).length).toBeGreaterThan(0);
    }
  });

  it("accepts a sha256 and rejects an uppercase one", () => {
    expect(validateMeta({ ...valid, sha256: "a".repeat(64) })).toEqual([]);
    expect(validateMeta({ ...valid, sha256: "A".repeat(64) }).length).toBeGreaterThan(0);
  });

  it("checks input.shape against imgsz (rule JSON Schema cannot express)", () => {
    const input = { ...(valid.input as object), shape: [1, 3, 320, 320] };
    expect(validateMeta({ ...valid, input })).toEqual([expect.stringContaining("does not match imgsz")]);
  });

  it("accepts the metas that the real exporter produced (fixtures from Python)", () => {
    for (const task of ["detect", "segment"] as const) {
      const fixture = loadFixture(task);
      expect(validateMeta(fixture.meta)).toEqual([]);
      expect(validateMeta(fixture.raw_meta)).toEqual([]);
    }
  });
});

describe("meta: helpers", () => {
  it("tells the two output layouts apart", () => {
    expect(outputLayout(parseMeta(read("meta.detect.valid.json")))).toBe("postprocessed");
    expect(outputLayout(parseMeta(read("meta.segment-raw.valid.json")))).toBe("raw");
  });

  it("lists output names in order", () => {
    expect(outputNames(parseMeta(read("meta.segment.valid.json")))).toEqual(["labels", "boxes", "scores", "masks"]);
  });

  it("looks up class names and falls back to the id", () => {
    const meta = loadFixture("detect").meta;
    expect(className(meta, 0)).toBe("cat");
    expect(className(meta, 1)).toBe("dog");
    expect(className(meta, 7)).toBe("7");
  });
});
