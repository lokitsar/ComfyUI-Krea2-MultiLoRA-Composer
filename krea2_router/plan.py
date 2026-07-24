from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters import LoadedRegionAdapters
from .config import Region, RouterConfig


@dataclass(frozen=True)
class RoutePlan:
    """Reusable routing state passed from the layout pass to mask refinement."""

    conditioning: Any
    prompt: str
    regions: tuple[Region, ...]
    loaded: tuple[LoadedRegionAdapters, ...]
    config: RouterConfig
    token_positions: tuple[tuple[int, ...], ...]
    used_concepts: tuple[str, ...]
    warnings: tuple[str, ...]
    width: int
    height: int


def sam3_box_prompt(plan: RoutePlan, character_index: int, padding: float = 0.0) -> tuple[dict, Region]:
    """Return one normalized center-format box accepted by ComfyUI-SAM3."""

    index = int(character_index) - 1
    if index < 0 or index >= len(plan.regions):
        raise ValueError(
            f"Character index {character_index} is unavailable; route plan has {len(plan.regions)} characters"
        )
    region = plan.regions[index]
    padding = max(0.0, min(0.5, float(padding)))
    grow_x = region.w * padding
    grow_y = region.h * padding
    x0 = max(0.0, region.x - grow_x)
    y0 = max(0.0, region.y - grow_y)
    x1 = min(1.0, region.x + region.w + grow_x)
    y1 = min(1.0, region.y + region.h + grow_y)
    box = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]
    return {"boxes": [box], "labels": [True]}, region
