from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Region

_COLORS = torch.tensor(
    [
        [0.95, 0.25, 0.30],
        [0.20, 0.55, 1.00],
        [0.20, 0.85, 0.45],
        [0.95, 0.70, 0.20],
        [0.65, 0.35, 0.95],
        [0.15, 0.85, 0.85],
        [0.95, 0.35, 0.75],
        [0.70, 0.80, 0.25],
    ],
    dtype=torch.float32,
)


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _inward_rect_mask(rows: int, cols: int, region: Region, feather: float) -> torch.Tensor:
    """Return a box mask with a soft edge wholly inside the box."""
    yy = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    xx = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    x0, y0, x1, y1 = region.box
    inside = (grid_x >= x0) & (grid_x <= x1) & (grid_y >= y0) & (grid_y <= y1)
    if feather <= 0.0:
        return inside.float()

    feather_x = max(1.0 / cols, region.w * feather)
    feather_y = max(1.0 / rows, region.h * feather)
    edge_x = torch.minimum(grid_x - x0, x1 - grid_x) / feather_x
    edge_y = torch.minimum(grid_y - y0, y1 - grid_y) / feather_y
    return (_smoothstep(torch.minimum(edge_x, edge_y)) * inside).float()


def build_region_masks(
    rows: int,
    cols: int,
    regions: list[Region],
    feather: float,
    overlap_policy: str = "nearest",
) -> torch.Tensor:
    """Build ``[regions, rows, cols]`` masks with deterministic overlap handling."""
    if rows < 1 or cols < 1:
        raise ValueError("mask grid must be at least 1x1")
    if not regions:
        return torch.zeros((0, rows, cols), dtype=torch.float32)

    masks = torch.stack([_inward_rect_mask(rows, cols, region, feather) for region in regions])
    return resolve_mask_overlaps(masks, regions, overlap_policy)


def build_routing_masks(
    rows: int,
    cols: int,
    regions: list[Region],
    feather: float,
    overlap_policy: str = "nearest",
    mask_modes: list[str] | None = None,
    exclusion_regions: list[Region] | None = None,
) -> torch.Tensor:
    """Build ordinary, unboxed-complement, and global masks in routing order."""
    if rows < 1 or cols < 1:
        raise ValueError("mask grid must be at least 1x1")
    if not regions:
        return torch.zeros((0, rows, cols), dtype=torch.float32)

    modes = mask_modes or ["region"] * len(regions)
    if len(modes) != len(regions):
        raise ValueError("mask_modes must match the number of routing regions")
    if any(mode not in {"region", "unboxed", "global"} for mode in modes):
        raise ValueError("mask_modes may only contain region, unboxed, or global")

    routed = torch.zeros((len(regions), rows, cols), dtype=torch.float32)
    region_indices = [index for index, mode in enumerate(modes) if mode == "region"]
    if region_indices:
        region_masks = build_region_masks(
            rows,
            cols,
            [regions[index] for index in region_indices],
            feather,
            overlap_policy,
        )
        for mask_index, region_index in enumerate(region_indices):
            routed[region_index] = region_masks[mask_index]

    exclusions = exclusion_regions
    if exclusions is None:
        exclusions = [regions[index] for index in region_indices]
    if exclusions:
        exclusion_masks = build_region_masks(rows, cols, exclusions, feather, "allow")
        occupied = exclusion_masks.amax(dim=0).clamp(0.0, 1.0)
    else:
        occupied = torch.zeros((rows, cols), dtype=torch.float32)

    for index, mode in enumerate(modes):
        if mode == "unboxed":
            routed[index] = 1.0 - occupied
        elif mode == "global":
            routed[index] = 1.0
    return routed


def resolve_mask_overlaps(
    masks: torch.Tensor,
    regions: list[Region] | tuple[Region, ...],
    overlap_policy: str = "nearest",
) -> torch.Tensor:
    """Apply the router overlap policy to rectangular or externally supplied masks."""

    if masks.ndim != 3:
        raise ValueError("masks must have shape [regions, rows, cols]")
    if masks.shape[0] != len(regions):
        raise ValueError(f"mask batch has {masks.shape[0]} masks but router has {len(regions)} regions")
    masks = masks.clone()
    if overlap_policy == "allow":
        return masks
    if overlap_policy == "normalize":
        total = masks.sum(dim=0, keepdim=True)
        return torch.where(total > 1.0, masks / total.clamp_min(1e-8), masks)
    if overlap_policy != "nearest":
        raise ValueError(f"unknown overlap policy: {overlap_policy}")

    active = masks > 0
    overlap = active.sum(dim=0) > 1
    if not torch.any(overlap):
        return masks

    rows, cols = masks.shape[-2:]
    yy = (torch.arange(rows, device=masks.device, dtype=torch.float32) + 0.5) / rows
    xx = (torch.arange(cols, device=masks.device, dtype=torch.float32) + 0.5) / cols
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    distances = []
    for region in regions:
        cx, cy = region.center
        distances.append((grid_x - cx).square() + (grid_y - cy).square())
    distance_stack = torch.stack(distances).masked_fill(~active, float("inf"))
    winners = distance_stack.argmin(dim=0)
    for index in range(len(regions)):
        masks[index] = torch.where(overlap & (winners != index), 0.0, masks[index])
    return masks


def resize_mask_batch(
    masks: torch.Tensor,
    rows: int,
    cols: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Resize a mask batch while retaining soft boundaries."""

    target = masks.to(device=device or masks.device, dtype=dtype or masks.dtype)
    return F.interpolate(target.unsqueeze(1), size=(rows, cols), mode="bilinear", align_corners=False).squeeze(1)


def masks_to_preview(masks: torch.Tensor) -> torch.Tensor:
    """Convert region masks to a ComfyUI IMAGE tensor ``[1,H,W,3]``."""
    if masks.ndim != 3:
        raise ValueError("masks must have shape [regions, rows, cols]")
    _, rows, cols = masks.shape
    preview = torch.full((rows, cols, 3), 0.035, dtype=torch.float32)
    for index, mask in enumerate(masks):
        color = _COLORS[index % len(_COLORS)]
        alpha = (mask * 0.82).unsqueeze(-1)
        preview = preview * (1.0 - alpha) + color * alpha
    return preview.clamp(0.0, 1.0).unsqueeze(0)
