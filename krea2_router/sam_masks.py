from __future__ import annotations

import torch

from .config import Region


def region_boxes_xyxy(regions: list[Region] | tuple[Region, ...], width: int, height: int) -> torch.Tensor:
    """Convert normalized router boxes to pixel-space corner boxes."""

    return torch.tensor(
        [
            [region.x * width, region.y * height, (region.x + region.w) * width, (region.y + region.h) * height]
            for region in regions
        ],
        dtype=torch.float32,
    )


def normalize_detection_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Accept SAM boxes in either normalized or pixel xyxy coordinates."""

    result = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).cpu()
    if result.numel() and float(result.abs().max()) <= 2.0:
        scale = torch.tensor([width, height, width, height], dtype=result.dtype)
        result = result * scale
    return result


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two xyxy box batches."""

    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    size = (bottom_right - top_left).clamp_min(0.0)
    intersection = size[..., 0] * size[..., 1]
    left_area = ((left[:, 2] - left[:, 0]).clamp_min(0.0) * (left[:, 3] - left[:, 1]).clamp_min(0.0))[:, None]
    right_area = ((right[:, 2] - right[:, 0]).clamp_min(0.0) * (right[:, 3] - right[:, 1]).clamp_min(0.0))[None, :]
    return intersection / (left_area + right_area - intersection).clamp_min(1e-8)


def assign_detections_to_regions(
    detection_boxes: torch.Tensor,
    regions: list[Region] | tuple[Region, ...],
    width: int,
    height: int,
    scores: torch.Tensor | None = None,
) -> tuple[list[int], list[float]]:
    """Greedily assign unique SAM detections to saved regions by spatial IoU."""

    detections = normalize_detection_boxes(detection_boxes, width, height)
    if detections.shape[0] < len(regions):
        raise ValueError(f"SAM3 returned {detections.shape[0]} objects for {len(regions)} character regions")
    targets = region_boxes_xyxy(regions, width, height)
    overlaps = box_iou(targets, detections)
    score_values = (
        torch.as_tensor(scores, dtype=torch.float32).flatten().cpu()
        if scores is not None
        else torch.zeros(detections.shape[0], dtype=torch.float32)
    )
    available = set(range(detections.shape[0]))
    assigned: list[int] = []
    assigned_iou: list[float] = []
    for region_index in range(len(regions)):
        ranked = sorted(
            available,
            key=lambda index: (float(overlaps[region_index, index]), float(score_values[index])),
            reverse=True,
        )
        winner = ranked[0]
        available.remove(winner)
        assigned.append(winner)
        assigned_iou.append(float(overlaps[region_index, winner]))
    return assigned, assigned_iou
