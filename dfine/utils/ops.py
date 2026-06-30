"""
Geometric utilities: box scaling, clipping, format conversion.
All functions operate on torch.Tensor unless noted.
"""
from __future__ import annotations

import torch


def scale_boxes(boxes, from_shape: tuple, to_shape: tuple):
    """
    Scale xyxy boxes from one image shape to another.

    Args:
        boxes:       Tensor [N, 4] xyxy
        from_shape:  (H, W) the boxes currently refer to
        to_shape:    (H, W) target image dimensions
    Returns:
        Tensor [N, 4] scaled xyxy
    """
    fh, fw = from_shape
    th, tw = to_shape
    scale = torch.tensor([tw / fw, th / fh, tw / fw, th / fh],
                         dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def clip_boxes(boxes, shape: tuple):
    """Clip xyxy boxes to image boundaries (H, W)."""
    h, w = shape
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h)
    return boxes


def xyxy_to_xywh(boxes):
    """Convert [x1,y1,x2,y2] → [cx,cy,w,h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def xywh_to_xyxy(boxes):
    """Convert [cx,cy,w,h] → [x1,y1,x2,y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)


def resolve_device(device):
    """Thin re-export so utils.ops can be imported standalone."""
    from dfine.utils.device import resolve_device as _r
    return _r(device)
