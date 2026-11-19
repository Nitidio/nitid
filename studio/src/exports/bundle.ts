/**
 * Orchestration: turns `ExportResults` (+ the source image) into downloadable
 * files. Everything heavy is loaded on demand: `fflate` (ZIP) via dynamic
 * import, the PDF writer via dynamic import.
 */
import { formatCsv, type CsvOptions } from "./csv";
import { formatJson } from "./json";
import type { BundleFormat, ExportFormat, ExportResults } from "./model";
import { baseName, classNameList, IMAGE_FORMATS } from "./model";
import { annotatedBlob, browserRaster, renderCrops, type RasterBackend } from "./raster";
import { formatClassesTxt, formatYoloTxt, type YoloOptions } from "./yolo";

export interface ExportFile {
  /** Path inside a ZIP, or the download file name. */
  name: string;
  blob: Blob;
}

export interface ExportContext {
  /** Anything `CanvasRenderingContext2D.drawImage` accepts (ImageBitmap, canvas, <img>). Required for png/crops/pdf/zip. */
  image?: unknown;
  raster?: RasterBackend;
  csv?: CsvOptions;
  yolo?: YoloOptions;
}

const MIME = {
  json: "application/json",
  csv: "text/csv;charset=utf-8",
  txt: "text/plain;charset=utf-8",
  png: "image/png",
  pdf: "application/pdf",
  zip: "application/zip",
} as const;

export function needsImage(format: ExportFormat): boolean {
  return IMAGE_FORMATS.includes(format);
}

function requireImage(ctx: ExportContext, format: ExportFormat): unknown {
  if (ctx.image === undefined || ctx.image === null) {
    throw new Error(`The ${format} export needs the source image`);
  }
  return ctx.image;
}

async function pdfBytes(results: ExportResults, ctx: ExportContext): Promise<Uint8Array> {
  const image = requireImage(ctx, "pdf");
  const backend = ctx.raster ?? browserRaster;
  const [{ buildPdfReport }, jpeg] = await Promise.all([
    import("./pdf"),
    annotatedBlob(results, image, backend, "image/jpeg", { maskAlpha: 0.4 }),
  ]);
  return buildPdfReport({ results, jpeg: new Uint8Array(await jpeg.arrayBuffer()) });
}

/** Files of one non-ZIP format (crops and zip entries can be several). */
export async function buildFiles(format: BundleFormat, results: ExportResults, ctx: ExportContext = {}): Promise<ExportFile[]> {
  const base = baseName(results.image.name);
  const backend = ctx.raster ?? browserRaster;
  switch (format) {
    case "json":
      return [{ name: `${base}.json`, blob: new Blob([formatJson(results)], { type: MIME.json }) }];
    case "csv":
      return [{ name: `${base}.csv`, blob: new Blob([formatCsv(results, ctx.csv)], { type: MIME.csv }) }];
    case "yolo":
      return [
        { name: `${base}.txt`, blob: new Blob([formatYoloTxt(results, ctx.yolo)], { type: MIME.txt }) },
        { name: "classes.txt", blob: new Blob([formatClassesTxt(classNameList(results))], { type: MIME.txt }) },
      ];
    case "png": {
      const blob = await annotatedBlob(results, requireImage(ctx, format), backend);
      return [{ name: `${base}_annotated.png`, blob }];
    }
    case "crops": {
      const crops = await renderCrops(results, requireImage(ctx, format), backend);
      return crops.map((c) => ({ name: `crops/${c.name}`, blob: c.blob }));
    }
    case "pdf": {
      const bytes = await pdfBytes(results, ctx);
      return [{ name: `${base}_report.pdf`, blob: new Blob([bytes as BlobPart], { type: MIME.pdf }) }];
    }
  }
}

/** ZIP of the selected formats. PNG-like members are stored, text members deflated; timestamps are fixed to `createdAt`. */
export async function buildZip(
  formats: readonly BundleFormat[],
  results: ExportResults,
  ctx: ExportContext = {},
): Promise<ExportFile> {
  const { zipSync } = await import("fflate");
  const parsed = new Date(results.createdAt);
  const mtime = Number.isNaN(parsed.getTime()) || parsed.getFullYear() < 1980 ? new Date(Date.UTC(1980, 0, 1)) : parsed;
  const entries: Record<string, [Uint8Array, { level: 0 | 6; mtime: Date }]> = {};
  for (const format of formats) {
    for (const f of await buildFiles(format, results, ctx)) {
      const stored = /\.(png|pdf)$/i.test(f.name);
      entries[f.name] = [new Uint8Array(await f.blob.arrayBuffer()), { level: stored ? 0 : 6, mtime }];
    }
  }
  const bytes = zipSync(entries);
  return { name: `${baseName(results.image.name)}_export.zip`, blob: new Blob([bytes as BlobPart], { type: MIME.zip }) };
}
