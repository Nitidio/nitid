"""
Unit tests for keypoint augmentations, dataset loader utilities, and pose validation evaluator.
"""

from __future__ import annotations

import torch

from dfine.utils.augmentations import (
    horizontal_flip_keypoints,
    scale_translate_keypoints,
    stretch_resize_keypoints,
)
from dfine.validator import _restore_original_keypoints


def test_horizontal_flip_keypoints_coco() -> None:
    width = 640.0
    # 1 instance, 17 keypoints (x, y, v)
    # Nose (0), L_Eye (1), R_Eye (2), L_Ear (3), R_Ear (4), L_Shoulder (5), R_Shoulder (6)
    kpts = torch.zeros((1, 17, 3), dtype=torch.float32)
    kpts[0, 1] = torch.tensor([100.0, 50.0, 2.0])  # Left eye x=100
    kpts[0, 2] = torch.tensor([200.0, 50.0, 2.0])  # Right eye x=200

    flipped = horizontal_flip_keypoints(kpts, width=width)

    # After flip:
    # Right eye (index 2 in flipped output) should contain original Left eye coordinates flipped: 640 - 100 = 540
    # Left eye (index 1 in flipped output) should contain original Right eye coordinates flipped: 640 - 200 = 440
    assert torch.allclose(flipped[0, 2, 0], torch.tensor(540.0))
    assert torch.allclose(flipped[0, 1, 0], torch.tensor(440.0))


def test_stretch_resize_keypoints() -> None:
    kpts = torch.tensor([[[100.0, 200.0, 2.0]]], dtype=torch.float32)
    orig_w, orig_h = 1000.0, 500.0
    target_size = 640

    resized = stretch_resize_keypoints(kpts, orig_w, orig_h, target_size)

    expected_x = 100.0 * (640.0 / 1000.0)  # 64.0
    expected_y = 200.0 * (640.0 / 500.0)  # 256.0
    assert torch.allclose(resized[0, 0, 0], torch.tensor(expected_x))
    assert torch.allclose(resized[0, 0, 1], torch.tensor(expected_y))


def test_scale_translate_keypoints() -> None:
    kpts = torch.tensor([[[50.0, 50.0, 2.0]]], dtype=torch.float32)
    factor = 1.5
    left, top = 20.0, 10.0

    transformed = scale_translate_keypoints(kpts, factor, left, top)

    expected_x = 50.0 * 1.5 + 20.0  # 95.0
    expected_y = 50.0 * 1.5 + 10.0  # 85.0
    assert torch.allclose(transformed[0, 0, 0], torch.tensor(expected_x))
    assert torch.allclose(transformed[0, 0, 1], torch.tensor(expected_y))


def test_restore_original_keypoints() -> None:
    kpts = torch.tensor([[160.0, 320.0, 2.0]], dtype=torch.float32)  # [1, 3]
    orig_w, orig_h = 1280, 720
    imgsz = 640

    restored = _restore_original_keypoints(kpts, orig_w, orig_h, imgsz)

    # 160 * (1280 / 640) = 320, 320 * (720 / 640) = 360
    assert torch.allclose(restored[0, 0], torch.tensor(320.0))
    assert torch.allclose(restored[0, 1], torch.tensor(360.0))
