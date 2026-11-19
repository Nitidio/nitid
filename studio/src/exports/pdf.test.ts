import { describe, expect, it } from "vitest";
import { buildPdfReport, fitText, readJpegInfo, toWinAnsi } from "./pdf";
import { det, results, tinyJpeg } from "./test-fixtures";

const latin1 = (b: Uint8Array): string => Array.from(b, (c) => String.fromCharCode(c)).join("");
const pageCount = (s: string): number => (s.match(/\/Type \/Page /g) ?? []).length;
const declaredCount = (s: string): number => Number(/\/Type \/Pages \/Kids \[[^\]]*\] \/Count (\d+)/.exec(s)?.[1]);

describe("buildPdfReport", () => {
  it("writes a valid single-page report with metadata, image and table text", () => {
    const bytes = buildPdfReport({ results: results({ backend: "openvino", source: "server", device: "NPU" }), jpeg: tinyJpeg() });
    const s = latin1(bytes);
    expect(s.startsWith("%PDF-1.4\n")).toBe(true);
    expect(s.trimEnd().endsWith("%%EOF")).toBe(true);
    expect(pageCount(s)).toBe(1);
    expect(declaredCount(s)).toBe(1);
    expect(s).toContain("(Nitid app detection report) Tj");
    expect(s).toContain("(street view.JPG \\(200 x 100 px\\)) Tj");
    expect(s).toContain("(nitid1s \\(detect\\)) Tj");
    expect(s).toContain("(Server: openvino, device NPU) Tj");
    expect(s).toContain("preprocess 3.1 ms, inference 28.4 ms, postprocess 1.2 ms");
    expect(s).toContain("(person \\(0\\)) Tj");
    expect(s).toContain("(car \\(2\\)) Tj");
    expect(s).toContain("(Page 1 of 1) Tj");
    expect(s).toContain("/Subtype /Image /Width 16 /Height 12 /ColorSpace /DeviceRGB");
    expect(s).toContain("/Im1 Do");
  });

  it("xref offsets point at the objects", () => {
    const s = latin1(buildPdfReport({ results: results(), jpeg: tinyJpeg() }));
    const start = Number(/startxref\n(\d+)\n/.exec(s)?.[1]);
    expect(s.slice(start, start + 4)).toBe("xref");
    const lines = s.slice(start).split("\n");
    const count = Number(/0 (\d+)/.exec(lines[1] ?? "")?.[1]);
    for (let id = 1; id < count; id++) {
      const off = Number((lines[2 + id] ?? "").slice(0, 10));
      expect(s.slice(off, off + `${id} 0 obj`.length)).toBe(`${id} 0 obj`);
    }
  });

  it("works without an image and for empty results", () => {
    const s = latin1(buildPdfReport({ results: results({ detections: [] }) }));
    expect(pageCount(s)).toBe(1);
    expect(s).toContain("(No detections.) Tj");
    expect(s).not.toContain("/Im1");
    expect(s).toContain("(Detections:) Tj");
  });

  it("breaks the table over pages and repeats the header and page footer", () => {
    const many = Array.from({ length: 1000 }, (_, i) => det({ classId: i % 80, name: `obj${i}`, confidence: 0.5 + (i % 50) / 100 }));
    const s = latin1(buildPdfReport({ results: results({ detections: many }), jpeg: tinyJpeg() }));
    const pages = pageCount(s);
    expect(pages).toBeGreaterThan(15);
    expect(pages).toBeLessThan(40);
    expect(declaredCount(s)).toBe(pages);
    expect((s.match(/\(Class\) Tj/g) ?? []).length).toBe(pages);
    expect(s).toContain(`(Page ${pages} of ${pages}) Tj`);
    expect(s).toContain("(obj999 \\(39\\)) Tj");
    expect((s.match(/\/Im1 Do/g) ?? []).length).toBe(1);
  });

  it("encodes Latin-1 as octal escapes and replaces unsupported scripts with '?'", () => {
    const s = latin1(buildPdfReport({ results: results({ detections: [det({ name: "señal 停" })] }) }));
    expect(s).toContain("(se\\361al ?");
    // The file body (excluding image bytes) stays ASCII.
    expect(/[\u0080-\uffff]/.test(s.replace(/^%PDF-1.4\n[^\n]*\n/, ""))).toBe(false);
  });

  it("rejects non-JPEG data", () => {
    expect(() => buildPdfReport({ results: results(), jpeg: new Uint8Array([1, 2, 3]) })).toThrow(/JPEG/);
  });
});

describe("helpers", () => {
  it("reads JPEG size and components", () => {
    expect(readJpegInfo(tinyJpeg())).toEqual({ width: 16, height: 12, components: 3 });
  });
  it("maps text to WinAnsi", () => {
    expect(toWinAnsi("a€bé停\n")).toEqual([97, 0x80, 98, 0xe9, 63, 32]);
  });
  it("truncates with dots", () => {
    expect(String.fromCharCode(...fitText("Wwwwwwwwwwwwwwwwwwww", 9, 40))).toMatch(/^W.*\.\.\.$/);
    expect(String.fromCharCode(...fitText("ok", 9, 40))).toBe("ok");
  });
});
