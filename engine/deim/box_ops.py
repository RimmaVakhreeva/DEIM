"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    w_clamped = w.clamp(min=0.0)
    h_clamped = h.clamp(min=0.0)
    x1 = x_c - 0.5 * w_clamped
    y1 = y_c - 0.5 * h_clamped
    x2 = x_c + 0.5 * w_clamped
    y2 = y_c + 0.5 * h_clamped
    
    # Ensure x2 >= x1 and y2 >= y1 to avoid invalid boxes
    x1 = torch.clamp(x1, min=0.0, max=1.0)
    y1 = torch.clamp(y1, min=0.0, max=1.0)
    x2 = torch.clamp(x2, min=0.0, max=1.0)
    y2 = torch.clamp(y2, min=0.0, max=1.0)
    
    # Final check to ensure x2 >= x1 and y2 >= y1
    x2 = torch.maximum(x2, x1)
    y2 = torch.maximum(y2, y1)
    
    b = [x1, y1, x2, y2]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # Handle degenerate boxes gracefully instead of crashing
    eps = 1e-7
    
    # Fix invalid boxes where x2 < x1 or y2 < y1 using functional operations
    # to avoid in-place modifications that break gradient computation
    
    # For boxes1: ensure x2 >= x1 and y2 >= y1
    x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(-1)
    x2_1_fixed = torch.maximum(x2_1, x1_1 + eps)
    y2_1_fixed = torch.maximum(y2_1, y1_1 + eps)
    boxes1_fixed = torch.stack([x1_1, y1_1, x2_1_fixed, y2_1_fixed], dim=-1)
    
    # For boxes2: ensure x2 >= x1 and y2 >= y1
    x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(-1)
    x2_2_fixed = torch.maximum(x2_2, x1_2 + eps)
    y2_2_fixed = torch.maximum(y2_2, y1_2 + eps)
    boxes2_fixed = torch.stack([x1_2, y1_2, x2_2_fixed, y2_2_fixed], dim=-1)
    
    # Now compute IoU with the fixed boxes
    iou, union = box_iou(boxes1_fixed, boxes2_fixed)

    lt = torch.min(boxes1_fixed[:, None, :2], boxes2_fixed[:, :2])
    rb = torch.max(boxes1_fixed[:, None, 2:], boxes2_fixed[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # Avoid division by zero
    giou = iou - (area - union) / (area + eps)
    
    # Handle any remaining NaN values
    giou = torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return giou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)