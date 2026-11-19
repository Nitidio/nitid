import type { ResultDetection, ResultItem, ViewFilter } from "./results-model";

export const DEFAULT_MIN_CONFIDENCE = 0.25;

export function defaultFilter(minConfidence = DEFAULT_MIN_CONFIDENCE): ViewFilter {
  return { minConfidence, hiddenClasses: new Set() };
}

/** Detections that pass the confidence threshold (inclusive) and the class filter. */
export function filterDetections(
  detections: readonly ResultDetection[],
  filter: ViewFilter,
): ResultDetection[] {
  return detections.filter(
    (d) => d.confidence >= filter.minConfidence && !filter.hiddenClasses.has(d.classId),
  );
}

export interface ClassSummary {
  classId: number;
  name: string;
  /** Detections of this class above the confidence threshold (ignores the class filter). */
  count: number;
  hidden: boolean;
}

/**
 * One row per class present in `items`, sorted by count (desc) then name. Counts
 * respect the confidence threshold but not the class filter, so a hidden class still
 * shows how many detections it would bring back.
 */
export function summarizeClasses(items: readonly ResultItem[], filter: ViewFilter): ClassSummary[] {
  const byId = new Map<number, ClassSummary>();
  for (const item of items) {
    for (const d of item.detections) {
      const row = byId.get(d.classId) ?? {
        classId: d.classId,
        name: d.name,
        count: 0,
        hidden: filter.hiddenClasses.has(d.classId),
      };
      if (d.confidence >= filter.minConfidence) row.count += 1;
      byId.set(d.classId, row);
    }
  }
  return [...byId.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/** Immutable toggle of one class. */
export function toggleClass(filter: ViewFilter, classId: number): ViewFilter {
  const hidden = new Set(filter.hiddenClasses);
  if (hidden.has(classId)) hidden.delete(classId);
  else hidden.add(classId);
  return { ...filter, hiddenClasses: hidden };
}

export function setAllClasses(filter: ViewFilter, classIds: readonly number[], visible: boolean): ViewFilter {
  return { ...filter, hiddenClasses: visible ? new Set() : new Set(classIds) };
}

/** Clamp a slider value into [min, 1]; non-finite values fall back to `min`. */
export function clampConfidence(value: number, min = 0): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(1, Math.max(min, value));
}
