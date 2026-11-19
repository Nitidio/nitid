/**
 * JSON export, schema "nitid-studio-results" version 1 (documented field by
 * field in docs/studio-development.md, "Exports").
 *
 * Keys are snake_case like the REST API. Numbers are rounded (pixels: 2
 * decimals, confidence: 6). `mask.polygon` is the outer contour of the
 * largest mask component in ORIGINAL image pixels (see `contour.ts`).
 */
import { maskToPolygon } from "./contour";
import type { ExportResults } from "./model";

export const JSON_SCHEMA_ID = "nitid-studio-results";
export const JSON_SCHEMA_VERSION = 1;

export interface JsonDetection {
  index: number;
  class_id: number;
  name: string;
  confidence: number;
  box: { x1: number; y1: number; x2: number; y2: number };
  has_mask: boolean;
  mask?: { polygon: Array<[number, number]>; pixels: number };
}

export interface JsonDocument {
  schema: typeof JSON_SCHEMA_ID;
  version: typeof JSON_SCHEMA_VERSION;
  created_at: string;
  source: "browser" | "server";
  image: { name: string; width: number; height: number };
  model: { id: string; task: "detect" | "segment" };
  runtime: { backend: string; device: string | null };
  timings_ms: { preprocess: number; inference: number; postprocess: number } | null;
  detection_count: number;
  detections: JsonDetection[];
}

const r = (v: number, digits: number): number => Number(v.toFixed(digits));

export function toJsonDocument(results: ExportResults): JsonDocument {
  return {
    schema: JSON_SCHEMA_ID,
    version: JSON_SCHEMA_VERSION,
    created_at: results.createdAt,
    source: results.source,
    image: { ...results.image },
    model: { id: results.model, task: results.task },
    runtime: { backend: results.backend, device: results.device },
    timings_ms: results.timings
      ? {
          preprocess: r(results.timings.preprocessMs, 3),
          inference: r(results.timings.inferenceMs, 3),
          postprocess: r(results.timings.postprocessMs, 3),
        }
      : null,
    detection_count: results.detections.length,
    detections: results.detections.map((d, index) => {
      const out: JsonDetection = {
        index,
        class_id: d.classId,
        name: d.name,
        confidence: r(d.confidence, 6),
        box: { x1: r(d.box.x1, 2), y1: r(d.box.y1, 2), x2: r(d.box.x2, 2), y2: r(d.box.y2, 2) },
        has_mask: d.hasMask,
      };
      if (d.mask) {
        const poly = maskToPolygon(d.mask);
        let pixels = 0;
        for (let i = 0; i < d.mask.data.length; i++) if (d.mask.data[i]) pixels++;
        out.mask = { polygon: poly ? poly.map(([x, y]) => [r(x, 2), r(y, 2)]) : [], pixels };
      }
      return out;
    }),
  };
}

export function formatJson(results: ExportResults): string {
  return `${JSON.stringify(toJsonDocument(results), null, 2)}\n`;
}
