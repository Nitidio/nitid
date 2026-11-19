import { describe, expect, it } from "vitest";
import { CSV_COLUMNS, formatCsv } from "./csv";
import { det, rectMask, results } from "./test-fixtures";

describe("formatCsv", () => {
  it("matches the golden output (CRLF, one row per detection)", () => {
    expect(formatCsv(results())).toBe(
      [
        "image,index,class_id,class_name,confidence,x1,y1,x2,y2,width,height,has_mask,mask_pixels",
        "street view.JPG,0,0,person,0.9,10,20,110,70,100,50,false,",
        "street view.JPG,1,2,car,0.5,100,0,200,50,100,50,false,",
        "",
      ].join("\r\n"),
    );
  });

  it("writes only the header for empty results", () => {
    expect(formatCsv(results({ detections: [] }))).toBe(`${CSV_COLUMNS.join(",")}\r\n`);
  });

  it("escapes commas, quotes and line breaks (RFC 4180)", () => {
    const r = results({
      image: { name: 'a,"b".png', width: 10, height: 10 },
      detections: [det({ name: 'say "hi", ok\nline2' })],
    });
    const out = formatCsv(r);
    expect(out).toContain('"a,""b"".png"');
    expect(out).toContain('"say ""hi"", ok\nline2"');
  });

  it("neutralises spreadsheet formulas in text cells only", () => {
    const r = results({ detections: [det({ name: "=HYPERLINK(1)" }), det({ name: "-x" }), det({ name: "@a" }), det({ name: "+1" })] });
    const rows = formatCsv(r).split("\r\n");
    expect(rows[1]).toContain(",'=HYPERLINK(1),");
    expect(rows[2]).toContain(",'-x,");
    expect(rows[3]).toContain(",'@a,");
    expect(rows[4]).toContain(",'+1,");
  });

  it("keeps Unicode class names and supports BOM and ';' delimiter", () => {
    const r = results({ detections: [det({ name: "señal; 停止" })] });
    const out = formatCsv(r, { bom: true, delimiter: ";" });
    expect(out.startsWith("﻿image;index;")).toBe(true);
    expect(out).toContain('"señal; 停止"');
  });

  it("reports mask pixel counts", () => {
    const r = results({ detections: [det({ hasMask: true, mask: rectMask(0, 0, 3, 2) })] });
    expect(formatCsv(r).split("\r\n")[1]).toMatch(/,true,6$/);
  });
});
