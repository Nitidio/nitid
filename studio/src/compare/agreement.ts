/**
 * Agreement between two detection lists (pure functions).
 *
 * A detection of A matches one of B when the boxes overlap with IoU >= `iouThreshold` (default 0.5)
 * and, by default, have the same class id. Matching is one-to-one and greedy: all candidate pairs
 * are sorted by IoU (highest first; ties go to the lower index in A, then in B) and a pair is taken
 * when neither side is used yet. Greedy by IoU is what COCO-style evaluation does per class; it is
 * not optimal in contrived cases, which is fine for a visual comparison.
 */
import type { Box } from "../api/types";

export interface MatchInput {
  box: Box;
  classId: number;
}

export interface MatchOptions {
  /** Minimum IoU for a pair (inclusive). Default 0.5. */
  iouThreshold?: number;
  /** Require equal `classId`. Default true. */
  sameClass?: boolean;
}

export interface Match {
  /** Index in list A. */
  a: number;
  /** Index in list B. */
  b: number;
  iou: number;
}

export interface MatchResult {
  matches: Match[];
  onlyA: number[];
  onlyB: number[];
}

export const DEFAULT_IOU_THRESHOLD = 0.5;

/** Intersection over union of two xyxy boxes; 0 when either has no area. */
export function iou(p: Box, q: Box): number {
  const aw = Math.max(0, p.x2 - p.x1);
  const ah = Math.max(0, p.y2 - p.y1);
  const bw = Math.max(0, q.x2 - q.x1);
  const bh = Math.max(0, q.y2 - q.y1);
  const iw = Math.max(0, Math.min(p.x2, q.x2) - Math.max(p.x1, q.x1));
  const ih = Math.max(0, Math.min(p.y2, q.y2) - Math.max(p.y1, q.y1));
  const inter = iw * ih;
  const union = aw * ah + bw * bh - inter;
  return union > 0 ? inter / union : 0;
}

export function matchDetections(
  a: readonly MatchInput[],
  b: readonly MatchInput[],
  options: MatchOptions = {},
): MatchResult {
  const threshold = options.iouThreshold ?? DEFAULT_IOU_THRESHOLD;
  const sameClass = options.sameClass ?? true;

  const candidates: Match[] = [];
  a.forEach((da, i) => {
    b.forEach((db, j) => {
      if (sameClass && da.classId !== db.classId) return;
      const v = iou(da.box, db.box);
      if (v >= threshold && v > 0) candidates.push({ a: i, b: j, iou: v });
    });
  });
  candidates.sort((x, y) => y.iou - x.iou || x.a - y.a || x.b - y.b);

  const usedA = new Set<number>();
  const usedB = new Set<number>();
  const matches: Match[] = [];
  for (const c of candidates) {
    if (usedA.has(c.a) || usedB.has(c.b)) continue;
    usedA.add(c.a);
    usedB.add(c.b);
    matches.push(c);
  }
  matches.sort((x, y) => x.a - y.a);
  return {
    matches,
    onlyA: a.map((_, i) => i).filter((i) => !usedA.has(i)),
    onlyB: b.map((_, j) => j).filter((j) => !usedB.has(j)),
  };
}

export interface AgreementSummary extends MatchResult {
  countA: number;
  countB: number;
  matched: number;
  /** matched / (countA + countB - matched), i.e. matched over the union of detections; 1 when both are empty. */
  agreement: number;
  /** Mean IoU of the matched pairs; null when nothing matched. */
  meanIou: number | null;
}

export function summarizeAgreement(
  a: readonly MatchInput[],
  b: readonly MatchInput[],
  options: MatchOptions = {},
): AgreementSummary {
  const m = matchDetections(a, b, options);
  const matched = m.matches.length;
  const union = a.length + b.length - matched;
  return {
    ...m,
    countA: a.length,
    countB: b.length,
    matched,
    agreement: union === 0 ? 1 : matched / union,
    meanIou: matched === 0 ? null : m.matches.reduce((s, x) => s + x.iou, 0) / matched,
  };
}
