import { unzipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { buildFiles, buildZip, needsImage } from "./bundle";
import { formatCsv } from "./csv";
import { formatJson } from "./json";
import { det, fakeRaster, rectMask, results } from "./test-fixtures";
import { formatYoloTxt } from "./yolo";

const text = async (b: Blob): Promise<string> => new TextDecoder().decode(await b.arrayBuffer());
const rasterCtx = () => ({ image: "IMG", raster: fakeRaster().backend });

describe("buildFiles", () => {
  it("builds text formats without an image", async () => {
    const r = results();
    const [json] = await buildFiles("json", r);
    const [csv] = await buildFiles("csv", r, { csv: { bom: false } });
    const yolo = await buildFiles("yolo", r);
    expect(json?.name).toBe("street_view.json");
    expect(await text(json!.blob)).toBe(formatJson(r));
    expect(csv?.name).toBe("street_view.csv");
    expect(await text(csv!.blob)).toBe(formatCsv(r));
    expect(yolo.map((f) => f.name)).toEqual(["street_view.txt", "classes.txt"]);
    expect(await text(yolo[0]!.blob)).toBe(formatYoloTxt(r));
    expect(await text(yolo[1]!.blob)).toBe("person\nclass_1\ncar\n");
  });

  it("image formats need the image", async () => {
    expect(needsImage("png")).toBe(true);
    expect(needsImage("json")).toBe(false);
    await expect(buildFiles("png", results())).rejects.toThrow(/source image/);
    await expect(buildFiles("pdf", results())).rejects.toThrow(/source image/);
  });

  it("builds PNG, crops and PDF with the injected raster backend", async () => {
    const ctx = rasterCtx();
    const png = await buildFiles("png", results(), ctx);
    expect(png[0]?.name).toBe("street_view_annotated.png");
    expect(png[0]?.blob.type).toBe("image/png");
    const crops = await buildFiles("crops", results(), ctx);
    expect(crops.map((c) => c.name)).toEqual(["crops/street_view_0000_person.png", "crops/street_view_0001_car.png"]);
    const pdf = await buildFiles("pdf", results(), ctx);
    expect(pdf[0]?.name).toBe("street_view_report.pdf");
    const head = new Uint8Array(await pdf[0]!.blob.arrayBuffer()).slice(0, 8);
    expect(String.fromCharCode(...head)).toBe("%PDF-1.4");
  });
});

describe("buildZip", () => {
  it("round-trips the selected members", async () => {
    const r = results({ detections: [det({ hasMask: true, mask: rectMask(10, 20, 4, 4) })] });
    const zip = await buildZip(["json", "csv", "yolo", "png", "crops"], r, rasterCtx());
    expect(zip.name).toBe("street_view_export.zip");
    expect(zip.blob.type).toBe("application/zip");
    const files = unzipSync(new Uint8Array(await zip.blob.arrayBuffer()));
    expect(Object.keys(files).sort()).toEqual([
      "classes.txt",
      "crops/street_view_0000_person.png",
      "street_view.csv",
      "street_view.json",
      "street_view.txt",
      "street_view_annotated.png",
    ]);
    expect(new TextDecoder().decode(files["street_view.csv"])).toBe(formatCsv(r));
    expect(Array.from(files["street_view_annotated.png"]!)).toEqual([0x89, 0x50, 0x4e, 0x47]);
  });

  it("is deterministic for the same inputs (fixed timestamps)", async () => {
    const a = new Uint8Array(await (await buildZip(["json", "csv"], results())).blob.arrayBuffer());
    const b = new Uint8Array(await (await buildZip(["json", "csv"], results())).blob.arrayBuffer());
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("works for empty results and a bad timestamp", async () => {
    const zip = await buildZip(["json", "csv", "yolo"], results({ detections: [], createdAt: "nope" }));
    const files = unzipSync(new Uint8Array(await zip.blob.arrayBuffer()));
    expect(new TextDecoder().decode(files["street_view.txt"])).toBe("");
    expect(new TextDecoder().decode(files["classes.txt"])).toBe("");
  });
});

describe("performance sanity", () => {
  it("formats 1000 detections (with masks) in well under a second each", async () => {
    const many = Array.from({ length: 1000 }, (_, i) =>
      det({
        classId: i % 80,
        name: `clase ${i} 停`,
        box: { x1: i % 100, y1: i % 50, x2: (i % 100) + 40, y2: (i % 50) + 30 },
        hasMask: true,
        mask: rectMask(i % 100, i % 50, 40, 30),
      }),
    );
    const r = results({ image: { name: "big.png", width: 1920, height: 1080 }, detections: many });
    const t0 = performance.now();
    formatCsv(r);
    formatJson(r);
    formatYoloTxt(r);
    const zip = await buildZip(["json", "csv", "yolo"], r);
    const ms = performance.now() - t0;
    expect(zip.blob.size).toBeGreaterThan(1000);
    expect(ms).toBeLessThan(5000); // generous bound for CI; measured ~100 ms locally
  });
});
