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


def prepare_external_masks(
    masks: torch.Tensor,
    expected: int,
    grow: int = 0,
    feather: int = 0,
) -> torch.Tensor:
    """Validate and softly expand/erode a ComfyUI MASK batch for identity routing."""

    if not torch.is_tensor(masks):
        raise TypeError("identity_masks must be a torch tensor")
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError("identity_masks must have shape [characters, height, width]")
    if masks.shape[0] != expected:
        raise ValueError(
            f"identity mask batch has {masks.shape[0]} masks but the route plan has {expected} characters; "
            "stack one SAM3 mask per character in Character A, B, C order"
        )
    result = torch.nan_to_num(masks.detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    radius = abs(int(grow))
    if radius:
        kernel = radius * 2 + 1
        source = result.unsqueeze(1)
        if grow > 0:
            source = F.max_pool2d(source, kernel, stride=1, padding=radius)
        else:
            source = 1.0 - F.max_pool2d(1.0 - source, kernel, stride=1, padding=radius)
        result = source.squeeze(1)

    blur_radius = max(0, int(feather))
    if blur_radius:
        sigma = max(0.5, blur_radius / 2.0)
        axis = torch.arange(-blur_radius, blur_radius + 1, device=result.device, dtype=result.dtype)
        kernel_1d = torch.exp(-(axis.square()) / (2.0 * sigma * sigma))
        kernel_1d /= kernel_1d.sum()
        source = F.pad(result.unsqueeze(1), (blur_radius,) * 4, mode="replicate")
        source = F.conv2d(source, kernel_1d.view(1, 1, 1, -1))
        source = F.conv2d(source, kernel_1d.view(1, 1, -1, 1))
        result = source.squeeze(1)
    return result.clamp(0.0, 1.0)


def resize_mask_batch(
    masks: torch.Tensor,
    rows: int,
    cols: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Resize a mask batch while retaining soft SAM boundaries."""

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
