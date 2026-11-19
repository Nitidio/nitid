import { boxToRect, classColor, labelPosition, labelText, scalePolygon } from "./geometry";
import type { ResultDetection } from "./results-model";

export interface DrawOptions {
  scale: number;
  imageWidth: number;
  imageHeight: number;
  showLabels: boolean;
  showMasks: boolean;
}

const LABEL_FONT_PX = 13;
const LABEL_PAD = 3;

/**
 * Draw detections onto a 2D context that already holds the image. Pure with respect to
 * everything but `ctx`, so it is tested with a recording fake.
 */
export function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: readonly ResultDetection[],
  opts: DrawOptions,
): void {
  const canvasWidth = opts.imageWidth * opts.scale;
  ctx.lineWidth = 2;
  ctx.font = `${LABEL_FONT_PX}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of detections) {
    const rect = boxToRect(d.box, opts.imageWidth, opts.imageHeight, opts.scale);
    if (!rect) continue;
    const color = classColor(d.classId);

    if (opts.showMasks && d.hasMask) {
      ctx.save();
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35;
      if (d.mask && d.mask.polygons.length > 0) {
        for (const poly of d.mask.polygons) {
          const pts = scalePolygon(poly, opts.scale);
          if (pts.length < 6) continue;
          ctx.beginPath();
          ctx.moveTo(pts[0] ?? 0, pts[1] ?? 0);
          for (let i = 2; i + 1 < pts.length; i += 2) ctx.lineTo(pts[i] ?? 0, pts[i + 1] ?? 0);
          ctx.closePath();
          ctx.fill();
        }
      } else {
        // API v1 does not serve mask pixels: mark the instance with a lighter box fill.
        ctx.globalAlpha = 0.18;
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      }
      ctx.restore();
    }

    ctx.strokeStyle = color;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

    if (opts.showLabels) {
      const text = labelText(d);
      const width = ctx.measureText(text).width + LABEL_PAD * 2;
      const height = LABEL_FONT_PX + LABEL_PAD * 2;
      const pos = labelPosition(rect, width, height, canvasWidth);
      ctx.fillStyle = color;
      ctx.fillRect(pos.x, pos.y, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, pos.x + LABEL_PAD, pos.y + LABEL_PAD);
    }
  }
}
