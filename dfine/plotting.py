"""
Visualisation helpers. plot_results() draws boxes on an image.
"""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
]


def plot_results(result, conf: bool, labels: bool, line_width, font_size) -> np.ndarray:
    img = result.orig_img.copy()
    if result.semantic_mask is not None:
        colorized = result.semantic_mask.colorize()
        return cv2.addWeighted(colorized, 0.45, img, 0.55, 0)
    if result.boxes is None or len(result.boxes) == 0:
        return img

    lw = line_width or max(round(sum(img.shape[:2]) / 2 * 0.003), 2)
    fs = font_size or max(lw - 1, 1)

    if result.masks is not None:
        overlay = img.copy()
        for index, mask in enumerate(result.masks.data.detach().cpu().numpy()):
            cls_id = int(result.boxes.cls[index])
            color = PALETTE[cls_id % len(PALETTE)]
            overlay[mask.astype(bool)] = color
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[i].int().tolist()
        cls_id = int(result.boxes.cls[i])
        score = float(result.boxes.conf[i])
        track_ids = result.boxes.id
        track_id = int(track_ids[i]) if track_ids is not None else None
        color_index = track_id if track_id is not None and track_id >= 0 else cls_id
        color = PALETTE[color_index % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

        if labels or conf:
            name = result.names.get(cls_id, str(cls_id))
            if track_id is not None and track_id >= 0:
                name = f"{name} #{track_id}"
            text = f"{name} {score:.2f}" if conf else name
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, (255, 255, 255), 1
            )
    return img
