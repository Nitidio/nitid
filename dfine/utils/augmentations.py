"""Deterministic geometry-aware augmentations for D-FINE vision tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as F
from PIL import Image

# Smallest crop side, as a fraction of the input side, that ``random_iou_crop`` samples.
IOU_CROP_MIN_FRACTION = 0.3


def zoom_out_short_side_limit(output_size: int) -> int:
    """Smallest zoom-out canvas side that keeps every IoU crop at or above ``output_size``.

    The detection recipe stretches the IoU crop to ``output_size``. A crop is at least
    ``IOU_CROP_MIN_FRACTION`` of the canvas on each axis, so a canvas whose shorter side
    reaches this value never has to be upsampled; any extra resolution is discarded.
    """
    return int(np.ceil(output_size / IOU_CROP_MIN_FRACTION))


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved training augmentation settings (probabilities are in ``[0, 1]``)."""

    profile: Literal["legacy", "dfine", "deim"] = "legacy"
    enabled: bool = True
    fliplr: float = 0.5
    scale: float = 0.5
    translate: float = 0.1
    crop: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    mosaic: float = 0.0
    mixup: float = 0.0
    close_mosaic: int = 10
    photometric: float = 0.0
    zoomout: float = 0.0
    iou_crop: float = 0.0

    def validate(self) -> None:
        if self.profile not in {"legacy", "dfine", "deim"}:
            raise ValueError("augmentation profile must be legacy, dfine, or deim")
        for name in ("fliplr", "crop", "mosaic", "mixup", "photometric", "zoomout", "iou_crop"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range [0, 1]")
        for name in ("scale", "translate", "hsv_h", "hsv_s", "hsv_v"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.scale >= 1.0:
            raise ValueError("scale must be < 1.0")
        if self.translate > 1.0:
            raise ValueError("translate must be <= 1.0")
        if self.hsv_h > 0.5 or self.hsv_s > 1.0 or self.hsv_v > 1.0:
            raise ValueError("hsv_h must be <= 0.5 and hsv_s/hsv_v must be <= 1.0")
        if self.close_mosaic < 0:
            raise ValueError("close_mosaic must be >= 0")


def letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor]:
    """Resize without distortion and pad to a square, updating absolute xyxy boxes."""
    width, height = image.size
    ratio = min(size / width, size / height)
    resized_w, resized_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, (left, top))
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] = result[:, [0, 2]] * ratio + left
        result[:, [1, 3]] = result[:, [1, 3]] * ratio + top
    return canvas, result


def stretch_resize(
    image: Image.Image, boxes: torch.Tensor, size: int
) -> tuple[Image.Image, torch.Tensor]:
    """Resize to a square like upstream D-FINE, updating absolute xyxy boxes."""
    width, height = image.size
    scale_x, scale_y = size / width, size / height
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] *= scale_x
        result[:, [1, 3]] *= scale_y
    return image.resize((size, size), Image.Resampling.BILINEAR), result


def resize_masks(masks: torch.Tensor, size: int) -> torch.Tensor:
    """Resize ``[N,H,W]`` instance masks with nearest-neighbor sampling."""
    if masks.numel() == 0:
        return torch.zeros((0, size, size), dtype=torch.uint8)
    return torch_f.interpolate(masks[:, None].float(), size=(size, size), mode="nearest")[:, 0].to(
        torch.uint8
    )


def resize_semantic_mask(mask: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Resize an integer ``[H,W]`` class map using nearest-neighbor sampling."""
    output_size = (size, size) if isinstance(size, int) else size
    return torch_f.interpolate(mask[None, None].float(), size=output_size, mode="nearest")[0, 0].to(
        mask.dtype
    )


def horizontal_flip(image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
    """Flip an image and absolute xyxy boxes horizontally."""
    width, _ = image.size
    result = boxes.clone()
    if result.numel():
        x1 = width - result[:, 2]
        x2 = width - result[:, 0]
        result[:, 0], result[:, 2] = x1, x2
    return F.hflip(image), result


def horizontal_flip_masks(masks: torch.Tensor) -> torch.Tensor:
    """Flip instance masks horizontally."""
    return torch.flip(masks, dims=[2])


def scale_translate_instances(
    image: Image.Image,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scale_gain: float,
    translate_gain: float,
    rng: random.Random,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Scale and translate an image, boxes, and aligned instance masks."""
    width, height = image.size
    factor = rng.uniform(1.0 - scale_gain, 1.0 + scale_gain)
    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
    tx = round(rng.uniform(-translate_gain, translate_gain) * width)
    ty = round(rng.uniform(-translate_gain, translate_gain) * height)
    left, top = (width - new_w) // 2 + tx, (height - new_h) // 2 + ty
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, (left, top))

    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] = result[:, [0, 2]] * factor + left
        result[:, [1, 3]] = result[:, [1, 3]] * factor + top

    output_masks = torch.zeros((len(masks), height, width), dtype=torch.uint8)
    if masks.numel():
        resized_masks = torch_f.interpolate(
            masks[:, None].float(), size=(new_h, new_w), mode="nearest"
        )[:, 0].to(torch.uint8)
        dst_x1, dst_y1 = max(left, 0), max(top, 0)
        dst_x2, dst_y2 = min(left + new_w, width), min(top + new_h, height)
        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            src_x1, src_y1 = dst_x1 - left, dst_y1 - top
            src_x2, src_y2 = src_x1 + dst_x2 - dst_x1, src_y1 + dst_y2 - dst_y1
            output_masks[:, dst_y1:dst_y2, dst_x1:dst_x2] = resized_masks[
                :, src_y1:src_y2, src_x1:src_x2
            ]
    return canvas, _clip_boxes(result, width, height), output_masks


def scale_translate_semantic(
    image: Image.Image,
    mask: torch.Tensor,
    scale_gain: float,
    translate_gain: float,
    rng: random.Random,
    ignore_index: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor]:
    """Scale and translate an image and dense class map on a fixed canvas."""
    width, height = image.size
    factor = rng.uniform(1.0 - scale_gain, 1.0 + scale_gain)
    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
    tx = round(rng.uniform(-translate_gain, translate_gain) * width)
    ty = round(rng.uniform(-translate_gain, translate_gain) * height)
    left, top = (width - new_w) // 2 + tx, (height - new_h) // 2 + ty

    resized_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    output_image = Image.new("RGB", (width, height), fill)
    output_image.paste(resized_image, (left, top))
    resized_mask = resize_semantic_mask(mask, (new_h, new_w))
    output_mask = torch.full((height, width), ignore_index, dtype=mask.dtype)

    dst_x1, dst_y1 = max(left, 0), max(top, 0)
    dst_x2, dst_y2 = min(left + new_w, width), min(top + new_h, height)
    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        src_x1, src_y1 = dst_x1 - left, dst_y1 - top
        src_x2, src_y2 = src_x1 + dst_x2 - dst_x1, src_y1 + dst_y2 - dst_y1
        output_mask[dst_y1:dst_y2, dst_x1:dst_x2] = resized_mask[src_y1:src_y2, src_x1:src_x2]
    return output_image, output_mask


def random_crop_instances(
    image: Image.Image,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    gain: float,
    rng: random.Random,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop an image and aligned box/mask instances."""
    width, height = image.size
    left = round(rng.uniform(0.0, gain) * width)
    right = round(rng.uniform(0.0, gain) * width)
    top = round(rng.uniform(0.0, gain) * height)
    bottom = round(rng.uniform(0.0, gain) * height)
    if left + right >= width or top + bottom >= height:
        keep = torch.ones(len(boxes), dtype=torch.bool)
        return image, boxes, masks, keep
    result = boxes.clone()
    result[:, [0, 2]] -= left
    result[:, [1, 3]] -= top
    crop_w, crop_h = width - left - right, height - top - bottom
    result = _clip_boxes(result, crop_w, crop_h)
    keep = _valid_boxes(result)
    cropped_masks = masks[:, top : height - bottom, left : width - right]
    return (
        image.crop((left, top, width - right, height - bottom)),
        result[keep],
        cropped_masks[keep],
        keep,
    )


def random_crop_semantic(
    image: Image.Image,
    mask: torch.Tensor,
    gain: float,
    rng: random.Random,
) -> tuple[Image.Image, torch.Tensor]:
    """Apply the same random edge crop to an image and dense class map."""
    width, height = image.size
    left = round(rng.uniform(0.0, gain) * width)
    right = round(rng.uniform(0.0, gain) * width)
    top = round(rng.uniform(0.0, gain) * height)
    bottom = round(rng.uniform(0.0, gain) * height)
    if left + right >= width or top + bottom >= height:
        return image, mask
    return (
        image.crop((left, top, width - right, height - bottom)),
        mask[top : height - bottom, left : width - right],
    )


def scale_translate(
    image: Image.Image,
    boxes: torch.Tensor,
    scale_gain: float,
    translate_gain: float,
    rng: random.Random,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor]:
    """Apply random isotropic scaling and translation on a fixed-size canvas."""
    width, height = image.size
    factor = rng.uniform(1.0 - scale_gain, 1.0 + scale_gain)
    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
    tx = round(rng.uniform(-translate_gain, translate_gain) * width)
    ty = round(rng.uniform(-translate_gain, translate_gain) * height)
    left, top = (width - new_w) // 2 + tx, (height - new_h) // 2 + ty
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, (left, top))
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] = result[:, [0, 2]] * factor + left
        result[:, [1, 3]] = result[:, [1, 3]] * factor + top
    return canvas, _clip_boxes(result, width, height)


def random_crop(
    image: Image.Image, boxes: torch.Tensor, gain: float, rng: random.Random
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Crop up to ``gain`` from the image edges and return the surviving-box mask."""
    width, height = image.size
    left = round(rng.uniform(0.0, gain) * width)
    right = round(rng.uniform(0.0, gain) * width)
    top = round(rng.uniform(0.0, gain) * height)
    bottom = round(rng.uniform(0.0, gain) * height)
    if left + right >= width or top + bottom >= height:
        return image, boxes, torch.ones(len(boxes), dtype=torch.bool)
    result = boxes.clone()
    result[:, [0, 2]] -= left
    result[:, [1, 3]] -= top
    crop_w, crop_h = width - left - right, height - top - bottom
    result = _clip_boxes(result, crop_w, crop_h)
    keep = _valid_boxes(result)
    return image.crop((left, top, width - right, height - bottom)), result[keep], keep


def color_jitter_hsv(
    image: Image.Image, cfg: AugmentationConfig, rng: random.Random
) -> Image.Image:
    """Apply deterministic hue, saturation, and brightness jitter."""
    image = F.adjust_hue(image, rng.uniform(-cfg.hsv_h, cfg.hsv_h)) if cfg.hsv_h else image
    image = (
        F.adjust_saturation(image, rng.uniform(1 - cfg.hsv_s, 1 + cfg.hsv_s))
        if cfg.hsv_s
        else image
    )
    return (
        F.adjust_brightness(image, rng.uniform(1 - cfg.hsv_v, 1 + cfg.hsv_v))
        if cfg.hsv_v
        else image
    )


def random_photometric_distort(
    image: Image.Image,
    rng: random.Random,
    p: float = 0.5,
) -> Image.Image:
    """Approximate D-FINE/torchvision photometric distortion for PIL images."""
    if p <= 0.0 or rng.random() >= p:
        return image
    transforms = [
        lambda img: F.adjust_brightness(img, rng.uniform(0.875, 1.125)),
        lambda img: F.adjust_contrast(img, rng.uniform(0.5, 1.5)),
        lambda img: F.adjust_saturation(img, rng.uniform(0.5, 1.5)),
        lambda img: F.adjust_hue(img, rng.uniform(-0.05, 0.05)),
    ]
    rng.shuffle(transforms)
    for transform in transforms:
        image = transform(image)
    return image


def random_zoom_out(
    image: Image.Image,
    boxes: torch.Tensor,
    rng: random.Random,
    p: float = 1.0,
    fill: tuple[int, int, int] = (0, 0, 0),
    max_scale: float = 4.0,
    short_side_limit: int | None = None,
) -> tuple[Image.Image, torch.Tensor]:
    """Place the image on a larger canvas, matching D-FINE's zoom-out role.

    When ``short_side_limit`` is given, the image is first downscaled (keeping its aspect ratio)
    so the canvas's shorter side does not exceed ``short_side_limit``. Without that bound a
    high-resolution photo zoomed out by up to ``max_scale`` produces a canvas of up to
    ``max_scale**2`` times its pixels, which wastes memory and can trip Pillow's
    decompression-bomb guard in the following crop.
    """
    if p <= 0.0 or rng.random() >= p:
        return image, boxes
    width, height = image.size
    scale = rng.uniform(1.0, max_scale)
    if short_side_limit is not None and scale * min(width, height) > short_side_limit:
        factor = short_side_limit / (scale * min(width, height))
        resized_w, resized_h = max(1, round(width * factor)), max(1, round(height * factor))
        boxes = boxes.clone()
        if boxes.numel():
            boxes[:, [0, 2]] *= resized_w / width
            boxes[:, [1, 3]] *= resized_h / height
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        width, height = resized_w, resized_h
    new_w, new_h = max(width, round(width * scale)), max(height, round(height * scale))
    left = rng.randint(0, max(new_w - width, 0))
    top = rng.randint(0, max(new_h - height, 0))
    canvas = Image.new("RGB", (new_w, new_h), fill)
    canvas.paste(image, (left, top))
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] += left
        result[:, [1, 3]] += top
    return canvas, result


def random_iou_crop(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    rng: random.Random,
    p: float = 0.8,
    trials: int = 40,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """SSD/torchvision-style random IoU crop for detection boxes."""
    if p <= 0.0 or rng.random() >= p or boxes.numel() == 0:
        return image, boxes, labels
    width, height = image.size
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None]
    threshold = thresholds[rng.randrange(len(thresholds))]
    if threshold is None:
        return image, boxes, labels

    for _ in range(trials):
        crop_w = rng.uniform(IOU_CROP_MIN_FRACTION, 1.0) * width
        crop_h = rng.uniform(IOU_CROP_MIN_FRACTION, 1.0) * height
        aspect = crop_w / max(crop_h, 1e-6)
        if not 0.5 <= aspect <= 2.0:
            continue
        left = rng.uniform(0, width - crop_w)
        top = rng.uniform(0, height - crop_h)
        crop = torch.tensor([left, top, left + crop_w, top + crop_h], dtype=boxes.dtype)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = (
            (centers[:, 0] > crop[0])
            & (centers[:, 0] < crop[2])
            & (centers[:, 1] > crop[1])
            & (centers[:, 1] < crop[3])
        )
        if not keep.any():
            continue
        cropped_boxes = boxes[keep].clone()
        ious, _ = box_iou_xyxy(cropped_boxes, crop[None, :])
        if float(ious.max().item()) < threshold:
            continue
        cropped_boxes[:, [0, 2]] -= crop[0]
        cropped_boxes[:, [1, 3]] -= crop[1]
        cropped_boxes = _clip_boxes(cropped_boxes, round(crop_w), round(crop_h))
        kept_labels = labels[keep]
        valid = _valid_boxes(cropped_boxes)
        if valid.any():
            return (
                image.crop((int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))),
                cropped_boxes[valid],
                kept_labels[valid],
            )
    return image, boxes, labels


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU for absolute xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6), union


def sanitize(
    boxes: torch.Tensor, labels: torch.Tensor, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip boxes and remove empty/non-finite boxes with their corresponding labels."""
    boxes = _clip_boxes(boxes, width, height)
    keep = _valid_boxes(boxes) & torch.isfinite(boxes).all(dim=1)
    return boxes[keep], labels[keep]


def sanitize_instances(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip and filter boxes, labels, and instance masks together."""
    boxes = _clip_boxes(boxes, width, height)
    keep = _valid_boxes(boxes) & torch.isfinite(boxes).all(dim=1)
    if len(masks) != len(boxes):
        raise ValueError("instance masks must be aligned with boxes")
    return boxes[keep], labels[keep], masks[keep]


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a float tensor in [0, 1]."""
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def _clip_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]].clamp_(0, width)
        result[:, [1, 3]].clamp_(0, height)
    return result


def _valid_boxes(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
