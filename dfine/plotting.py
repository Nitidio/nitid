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


COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]

CROWDPOSE_SKELETON = [
    (12, 13),
    (13, 0),
    (13, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
]

KEYPOINT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
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

    if result.keypoints is not None and len(result.keypoints):
        kpts_xy = result.keypoints.xy.detach().cpu().numpy()
        kpts_conf = (
            result.keypoints.conf.detach().cpu().numpy()
            if result.keypoints.conf is not None
            else None
        )
        radius = max(lw, 3)

        for i in range(len(result.keypoints)):
            kpts = kpts_xy[i]
            conf_i = kpts_conf[i] if kpts_conf is not None else None
            num_kpts = len(kpts)

            skeleton = (
                COCO_SKELETON if num_kpts == 17 else (CROWDPOSE_SKELETON if num_kpts == 14 else [])
            )

            # Draw skeleton limbs
            for p1, p2 in skeleton:
                if p1 < num_kpts and p2 < num_kpts:
                    if conf_i is None or (conf_i[p1] > 0.3 and conf_i[p2] > 0.3):
                        pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                        pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                        limb_color = KEYPOINT_COLORS[p1 % len(KEYPOINT_COLORS)]
                        cv2.line(img, pt1, pt2, limb_color, max(1, lw - 1))

            # Draw keypoint dots
            for k_idx, (kx, ky) in enumerate(kpts):
                if conf_i is None or conf_i[k_idx] > 0.3:
                    kp_color = KEYPOINT_COLORS[k_idx % len(KEYPOINT_COLORS)]
                    cv2.circle(img, (int(kx), int(ky)), radius, kp_color, -1)

    return img
