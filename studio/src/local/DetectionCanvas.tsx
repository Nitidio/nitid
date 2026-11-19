import { useEffect, useRef } from "react";
import { drawDetections } from "./draw";
import { fitScale } from "./geometry";
import type { ResultDetection } from "./results-model";

/** The image and its overlay are drawn at most this wide (CSS makes it responsive). */
export const MAX_CANVAS_WIDTH = 1600;

export interface DetectionCanvasProps {
  image: ImageBitmap | null;
  width: number;
  height: number;
  detections: readonly ResultDetection[];
  showLabels: boolean;
  showMasks: boolean;
  /** Text alternative; the table next to the canvas carries the full data. */
  label: string;
}

export function DetectionCanvas({
  image,
  width,
  height,
  detections,
  showLabels,
  showMasks,
  label,
}: DetectionCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scale = fitScale(width, MAX_CANVAS_WIDTH);
  const cw = Math.max(1, Math.round(width * scale));
  const ch = Math.max(1, Math.round(height * scale));

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return; // jsdom has no 2D context
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image) ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    else {
      ctx.fillStyle = "#555";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    drawDetections(ctx, detections, {
      scale,
      imageWidth: width,
      imageHeight: height,
      showLabels,
      showMasks,
    });
  }, [image, detections, scale, width, height, showLabels, showMasks]);

  return <canvas ref={ref} width={cw} height={ch} className="detection-canvas" role="img" aria-label={label} />;
}
