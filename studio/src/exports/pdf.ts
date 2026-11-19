/**
 * Minimal PDF 1.4 writer for the detection report. No dependency.
 *
 * Why hand-written: the report needs one embedded JPEG, Helvetica text and a
 * table; that is ~250 lines, versus 300-600 kB (minified) for pdf-lib/jsPDF
 * that would ship to the browser for the same result.
 *
 * Limits (documented in docs/studio-development.md):
 *  - Text uses the built-in Helvetica / Helvetica-Bold with WinAnsiEncoding, so
 *    only Latin-1 (+ a few cp1252 punctuation marks) can be shown. Other
 *    characters (CJK, Cyrillic, ...) are printed as "?". The PDF is a human
 *    summary; JSON/CSV keep the exact Unicode names.
 *  - Content streams are not compressed (keeps the output greppable and
 *    testable); the JPEG is embedded as-is (DCTDecode).
 *  - The file is pure ASCII outside the JPEG stream.
 */
import type { ExportResults } from "./model";

const PAGE_W = 595.28;
const PAGE_H = 841.89;
const MARGIN = 40;
const ROW_H = 15;
const FONT = 9;

/** Helvetica advance widths (1/1000 em) for ASCII 32..126; other characters use 556. */
const HELV = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556, 556,
  556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722,
  778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556,
  278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260,
  334, 584,
];

const CP1252: Record<number, number> = {
  0x20ac: 0x80,
  0x2026: 0x85,
  0x2018: 0x91,
  0x2019: 0x92,
  0x201c: 0x93,
  0x201d: 0x94,
  0x2022: 0x95,
  0x2013: 0x96,
  0x2014: 0x97,
  0x2122: 0x99,
};

/** Maps text to WinAnsi byte values; unsupported characters become "?". */
export function toWinAnsi(text: string): number[] {
  const out: number[] = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    if (cp < 32 || cp === 127) out.push(32);
    else if (cp < 127 || (cp >= 0xa0 && cp <= 0xff)) out.push(cp);
    else out.push(CP1252[cp] ?? 63);
  }
  return out;
}

function textWidth(bytes: readonly number[], size: number): number {
  let w = 0;
  for (const b of bytes) w += b >= 32 && b <= 126 ? (HELV[b - 32] as number) : 556;
  return (w * size) / 1000;
}

/** Truncates to `maxWidth` points, adding "..." when cut. */
export function fitText(text: string, size: number, maxWidth: number): number[] {
  const bytes = toWinAnsi(text);
  if (textWidth(bytes, size) <= maxWidth) return bytes;
  const dots = [46, 46, 46];
  let n = bytes.length;
  while (n > 0 && textWidth([...bytes.slice(0, n), ...dots], size) > maxWidth) n--;
  return [...bytes.slice(0, n), ...dots];
}

/** PDF string literal, pure ASCII (non-ASCII bytes as octal escapes). */
function literal(bytes: readonly number[]): string {
  let s = "(";
  for (const b of bytes) {
    if (b === 0x28 || b === 0x29 || b === 0x5c) s += `\\${String.fromCharCode(b)}`;
    else if (b < 32 || b > 126) s += `\\${b.toString(8).padStart(3, "0")}`;
    else s += String.fromCharCode(b);
  }
  return `${s})`;
}

export interface JpegInfo {
  width: number;
  height: number;
  components: 1 | 3;
}

/** Reads size and component count from the first SOFn marker. Throws on non-JPEG / CMYK. */
export function readJpegInfo(bytes: Uint8Array): JpegInfo {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new Error("Not a JPEG");
  let i = 2;
  while (i + 9 < bytes.length) {
    if (bytes[i] !== 0xff) {
      i++;
      continue;
    }
    const marker = bytes[i + 1] as number;
    if (marker === 0xff) {
      i++;
      continue;
    }
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      const height = ((bytes[i + 5] as number) << 8) | (bytes[i + 6] as number);
      const width = ((bytes[i + 7] as number) << 8) | (bytes[i + 8] as number);
      const components = bytes[i + 9] as number;
      if (components !== 1 && components !== 3) throw new Error("Unsupported JPEG colour model");
      return { width, height, components };
    }
    const len = ((bytes[i + 2] as number) << 8) | (bytes[i + 3] as number);
    i += 2 + len;
  }
  throw new Error("JPEG has no frame header");
}

export interface PdfReportInput {
  results: ExportResults;
  /** Annotated image as JPEG bytes (optional: the report works without it). */
  jpeg?: Uint8Array;
}

class PdfBuilder {
  private chunks: Uint8Array[] = [];
  private length = 0;
  private offsets: number[] = [];

  private push(bytes: Uint8Array): void {
    this.chunks.push(bytes);
    this.length += bytes.length;
  }
  ascii(s: string): void {
    const b = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 0xff;
    this.push(b);
  }
  header(): void {
    this.ascii("%PDF-1.4\n%âãÏÓ\n");
  }
  /** Writes object `id` (1-based). */
  object(id: number, body: string, stream?: Uint8Array): void {
    this.offsets[id] = this.length;
    if (stream) {
      this.ascii(`${id} 0 obj\n<< ${body} /Length ${stream.length} >>\nstream\n`);
      this.push(stream);
      this.ascii("\nendstream\nendobj\n");
    } else {
      this.ascii(`${id} 0 obj\n${body}\nendobj\n`);
    }
  }
  finish(rootId: number, infoId: number, objectCount: number): Uint8Array {
    const xrefAt = this.length;
    let x = `xref\n0 ${objectCount + 1}\n0000000000 65535 f \n`;
    for (let id = 1; id <= objectCount; id++) x += `${String(this.offsets[id]).padStart(10, "0")} 00000 n \n`;
    x += `trailer\n<< /Size ${objectCount + 1} /Root ${rootId} 0 R /Info ${infoId} 0 R >>\nstartxref\n${xrefAt}\n%%EOF\n`;
    this.ascii(x);
    const out = new Uint8Array(this.length);
    let at = 0;
    for (const c of this.chunks) {
      out.set(c, at);
      at += c.length;
    }
    return out;
  }
}

interface PageContent {
  ops: string[];
  usesImage: boolean;
}

const n2 = (v: number): string => v.toFixed(2);

function pdfDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const p = (v: number): string => String(v).padStart(2, "0");
  return `D:${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}Z`;
}

/** Builds the report. Layout: page 1 = title, metadata, image, table start; table continues on later pages with a repeated header. */
export function buildPdfReport(input: PdfReportInput): Uint8Array {
  const { results, jpeg } = input;
  const info = jpeg ? readJpegInfo(jpeg) : null;
  const contentW = PAGE_W - 2 * MARGIN;
  const pages: PageContent[] = [];
  let page: PageContent = { ops: [], usesImage: false };
  pages.push(page);
  let y = PAGE_H - MARGIN;

  const text = (bytes: readonly number[], x: number, yy: number, size: number, bold = false): void => {
    page.ops.push(`BT /${bold ? "F2" : "F1"} ${size} Tf ${n2(x)} ${n2(yy)} Td ${literal(bytes)} Tj ET`);
  };
  const line = (s: string, x: number, yy: number, size: number, bold = false, maxW = contentW): void =>
    text(fitText(s, size, maxW), x, yy, size, bold);

  line("Nitid app detection report", MARGIN, y - 16, 18, true);
  y -= 34;

  const meta: Array<[string, string]> = [
    ["Image", `${results.image.name} (${results.image.width} x ${results.image.height} px)`],
    ["Model", `${results.model} (${results.task})`],
    ["Runtime", `${results.source === "browser" ? "Browser" : "Server"}: ${results.backend}${results.device ? `, device ${results.device}` : ""}`],
    [
      "Timing",
      results.timings
        ? `preprocess ${results.timings.preprocessMs.toFixed(1)} ms, inference ${results.timings.inferenceMs.toFixed(1)} ms, postprocess ${results.timings.postprocessMs.toFixed(1)} ms`
        : "not available",
    ],
    ["Created", results.createdAt],
    ["Detections", String(results.detections.length)],
  ];
  for (const [k, v] of meta) {
    line(`${k}:`, MARGIN, y - 10, 10, true, 70);
    line(v, MARGIN + 75, y - 10, 10, false, contentW - 75);
    y -= 14;
  }
  y -= 8;

  if (info) {
    const maxH = 300;
    const scale = Math.min(contentW / info.width, maxH / info.height);
    const w = info.width * scale;
    const h = info.height * scale;
    page.ops.push(`q ${n2(w)} 0 0 ${n2(h)} ${n2(MARGIN)} ${n2(y - h)} cm /Im1 Do Q`);
    page.usesImage = true;
    y -= h + 14;
  }

  // Table
  const cols: Array<{ title: string; x: number; w: number; right?: boolean }> = [
    { title: "#", x: 0, w: 28, right: true },
    { title: "Class", x: 36, w: 150 },
    { title: "Conf.", x: 190, w: 40, right: true },
    { title: "x1", x: 240, w: 45, right: true },
    { title: "y1", x: 295, w: 45, right: true },
    { title: "x2", x: 350, w: 45, right: true },
    { title: "y2", x: 405, w: 45, right: true },
    { title: "Mask", x: 465, w: contentW - 465 },
  ];
  const drawCell = (s: string, c: (typeof cols)[number], yy: number, bold: boolean): void => {
    const bytes = fitText(s, FONT, c.w);
    const x = MARGIN + c.x + (c.right ? c.w - textWidth(bytes, FONT) : 0);
    text(bytes, x, yy, FONT, bold);
  };
  const header = (): void => {
    for (const c of cols) drawCell(c.title, c, y - FONT, true);
    page.ops.push(`0.5 w ${n2(MARGIN)} ${n2(y - ROW_H + 3)} m ${n2(MARGIN + contentW)} ${n2(y - ROW_H + 3)} l S`);
    y -= ROW_H + 2;
  };
  const newPage = (): void => {
    page = { ops: [], usesImage: false };
    pages.push(page);
    y = PAGE_H - MARGIN;
  };

  if (results.detections.length === 0) {
    line("No detections.", MARGIN, y - 10, 10);
  } else {
    const roomFor = (rows: number): boolean => y - rows * ROW_H >= MARGIN + 20;
    if (!roomFor(3)) newPage();
    header();
    results.detections.forEach((d, i) => {
      if (!roomFor(1)) {
        newPage();
        header();
      }
      const vals = [
        String(i + 1),
        `${d.name} (${d.classId})`,
        d.confidence.toFixed(3),
        d.box.x1.toFixed(1),
        d.box.y1.toFixed(1),
        d.box.x2.toFixed(1),
        d.box.y2.toFixed(1),
        d.hasMask ? "yes" : "no",
      ];
      cols.forEach((c, k) => drawCell(vals[k] as string, c, y - FONT, false));
      y -= ROW_H;
    });
  }

  // Footers (need the final page count).
  pages.forEach((p, idx) => {
    const label = fitText(`Page ${idx + 1} of ${pages.length}`, 8, 200);
    p.ops.push(`BT /F1 8 Tf ${n2(PAGE_W - MARGIN - textWidth(label, 8))} ${n2(MARGIN - 20)} Td ${literal(label)} Tj ET`);
  });

  // Object numbering: 1 catalog, 2 pages, 3 F1, 4 F2, 5 info, 6 image (if any), then page+content pairs.
  const b = new PdfBuilder();
  b.header();
  const imageId = info ? 6 : 0;
  const firstPageId = info ? 7 : 6;
  const kids = pages.map((_, i) => `${firstPageId + 2 * i} 0 R`).join(" ");
  b.object(1, "<< /Type /Catalog /Pages 2 0 R >>");
  b.object(2, `<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  b.object(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
  b.object(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>");
  const created = pdfDate(results.createdAt);
  b.object(
    5,
    `<< /Title ${literal(toWinAnsi(`Nitid app report: ${results.image.name}`))} /Producer (Nitid app)${created ? ` /CreationDate (${created})` : ""} >>`,
  );
  if (info && jpeg) {
    b.object(
      imageId,
      `/Type /XObject /Subtype /Image /Width ${info.width} /Height ${info.height} /ColorSpace /${info.components === 3 ? "DeviceRGB" : "DeviceGray"} /BitsPerComponent 8 /Filter /DCTDecode`,
      jpeg,
    );
  }
  const encoder = new TextEncoder();
  pages.forEach((p, i) => {
    const pageId = firstPageId + 2 * i;
    const xobj = p.usesImage ? ` /XObject << /Im1 ${imageId} 0 R >>` : "";
    b.object(
      pageId,
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${PAGE_W} ${PAGE_H}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >>${xobj} >> /Contents ${pageId + 1} 0 R >>`,
    );
    b.object(pageId + 1, "", encoder.encode(p.ops.join("\n")));
  });
  return b.finish(1, 5, firstPageId + 2 * pages.length - 1);
}
