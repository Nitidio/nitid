"""Deterministic geometry-aware augmentations for D-FINE vision tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as F
from PIL import Image


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved training augmentation settings (probabilities are in ``[0, 1]``)."""

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

    def validate(self) -> None:
        for name in ("fliplr", "crop", "mosaic", "mixup"):
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


COCO_FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def horizontal_flip_keypoints(
    keypoints: torch.Tensor,
    width: float,
    flip_map: list[int] | None = None,
) -> torch.Tensor:
    """Flip keypoint x-coordinates and swap bilateral joint pairs."""
    if not keypoints.numel():
        return keypoints.clone()

    res = keypoints.clone()
    if res.dim() == 2 and res.shape[1] % 3 == 0:
        num_kpts = res.shape[1] // 3
        res = res.view(-1, num_kpts, 3)
    elif res.dim() == 2 and res.shape[1] % 2 == 0:
        num_kpts = res.shape[1] // 2
        res = res.view(-1, num_kpts, 2)

    # Flip x coordinates for keypoints
    res[..., 0] = width - res[..., 0]

    # Swap bilateral pairs if flip_map is provided
    map_to_use = (
        flip_map if flip_map is not None else (COCO_FLIP_MAP if res.shape[1] == 17 else None)
    )
    if map_to_use is not None and len(map_to_use) == res.shape[1]:
        res = res[:, map_to_use, :]

    return res.view_as(keypoints)


def stretch_resize_keypoints(
    keypoints: torch.Tensor, orig_w: float, orig_h: float, target_size: int
) -> torch.Tensor:
    """Scale keypoints to a square target size."""
    if not keypoints.numel():
        return keypoints.clone()

    scale_x, scale_y = target_size / max(1.0, orig_w), target_size / max(1.0, orig_h)
    res = keypoints.clone()
    if res.dim() == 2 and res.shape[1] % 3 == 0:
        num_kpts = res.shape[1] // 3
        res_view = res.view(-1, num_kpts, 3)
        res_view[..., 0] *= scale_x
        res_view[..., 1] *= scale_y
    elif res.dim() == 2 and res.shape[1] % 2 == 0:
        num_kpts = res.shape[1] // 2
        res_view = res.view(-1, num_kpts, 2)
        res_view[..., 0] *= scale_x
        res_view[..., 1] *= scale_y
    elif res.dim() == 3:
        res[..., 0] *= scale_x
        res[..., 1] *= scale_y
    return res


def scale_translate_keypoints(
    keypoints: torch.Tensor, factor: float, left: float, top: float
) -> torch.Tensor:
    """Scale and translate keypoints."""
    if not keypoints.numel():
        return keypoints.clone()

    res = keypoints.clone()
    if res.dim() == 2 and res.shape[1] % 3 == 0:
        num_kpts = res.shape[1] // 3
        res_view = res.view(-1, num_kpts, 3)
        res_view[..., 0] = res_view[..., 0] * factor + left
        res_view[..., 1] = res_view[..., 1] * factor + top
    elif res.dim() == 2 and res.shape[1] % 2 == 0:
        num_kpts = res.shape[1] // 2
        res_view = res.view(-1, num_kpts, 2)
        res_view[..., 0] = res_view[..., 0] * factor + left
        res_view[..., 1] = res_view[..., 1] * factor + top
    elif res.dim() == 3:
        res[..., 0] = res[..., 0] * factor + left
        res[..., 1] = res[..., 1] * factor + top
    return res
