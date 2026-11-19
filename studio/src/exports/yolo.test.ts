import { describe, expect, it } from "vitest";
import { formatClassesTxt, formatYoloTxt, parseYoloTxt } from "./yolo";
import { classNameList } from "./model";
import { det, maskFromRows, rectMask, results } from "./test-fixtures";

describe("formatYoloTxt", () => {
  it("writes normalised cx cy w h (golden)", () => {
    // image 200x100; box 10,20,110,70 -> cx 60/200=.3, cy 45/100=.45, w .5, h .5
    expect(formatYoloTxt(results())).toBe(
      ["0 0.300000 0.450000 0.500000 0.500000", "2 0.750000 0.250000 0.500000 0.500000", ""].join("\n"),
    );
  });

  it("gives an empty file when there are no detections", () => {
    expect(formatYoloTxt(results({ detections: [] }))).toBe("");
  });

  it("clamps boxes that leave the image", () => {
    const r = results({ detections: [det({ box: { x1: -20, y1: -5, x2: 250, y2: 120 } })] });
    expect(formatYoloTxt(r)).toBe("0 0.500000 0.500000 1.000000 1.000000\n");
  });

  it("writes polygon lines for masks and round-trips through the parser", () => {
    const mask = maskFromRows(20, 30, ["####", "####", "####"]); // 4x3 at (20,30) in a 200x100 image
    const r = results({
      task: "segment",
      detections: [det({ box: { x1: 20, y1: 30, x2: 24, y2: 33 }, hasMask: true, mask })],
    });
    const text = formatYoloTxt(r);
    const [line] = parseYoloTxt(text);
    expect(line?.kind).toBe("polygon");
    if (line?.kind !== "polygon") return;
    expect(line.classId).toBe(0);
    const px = line.points.map(([x, y]) => `${Math.round(x * 200)},${Math.round(y * 100)}`);
    expect(new Set(px)).toEqual(new Set(["20,30", "24,30", "24,33", "20,33"]));
  });

  it("can be forced to boxes, and falls back to a box when the mask is unusable", () => {
    const r = results({
      detections: [
        det({ hasMask: true, mask: rectMask(10, 20, 100, 50) }),
        det({ hasMask: true, mask: { x: 0, y: 0, width: 2, height: 2, data: new Uint8Array(4) } }),
      ],
    });
    expect(parseYoloTxt(formatYoloTxt(r, { segmentation: "box" })).map((l) => l.kind)).toEqual(["box", "box"]);
    expect(parseYoloTxt(formatYoloTxt(r)).map((l) => l.kind)).toEqual(["polygon", "box"]);
  });

  it("round-trips boxes within 6-decimal precision", () => {
    const r = results({
      image: { name: "a.png", width: 1920, height: 1080 },
      detections: [det({ box: { x1: 123.4, y1: 56.7, x2: 800.1, y2: 900.9 }, classId: 79 })],
    });
    const [l] = parseYoloTxt(formatYoloTxt(r));
    expect(l?.kind).toBe("box");
    if (l?.kind !== "box") return;
    expect(l.classId).toBe(79);
    expect((l.cx - l.w / 2) * 1920).toBeCloseTo(123.4, 2);
    expect((l.cy + l.h / 2) * 1080).toBeCloseTo(900.9, 2);
  });
});

describe("parseYoloTxt", () => {
  it("ignores blank lines and CRLF; rejects bad lines with the line number", () => {
    expect(parseYoloTxt("\r\n0 .5 .5 .1 .1\r\n\r\n")).toHaveLength(1);
    expect(() => parseYoloTxt("0 .5 .5 .1")).toThrow(/line 1/);
    expect(() => parseYoloTxt("0 .5 .5 .1 .1\nx 1 2 3 4")).toThrow(/line 2/);
    expect(() => parseYoloTxt("0 .1 .1 .2 .2 .3 .3 .4")).toThrow(SyntaxError);
  });
});

describe("classes.txt", () => {
  it("lists names by id with placeholders for gaps, Unicode intact", () => {
    const r = results({ detections: [det({ classId: 0, name: "persona" }), det({ classId: 3, name: "señal de tráfico 停" })] });
    expect(classNameList(r)).toEqual(["persona", "class_1", "class_2", "señal de tráfico 停"]);
    expect(formatClassesTxt(classNameList(r))).toBe("persona\nclass_1\nclass_2\nseñal de tráfico 停\n");
    expect(formatClassesTxt([])).toBe("");
  });
});
