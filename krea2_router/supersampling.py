from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SupersamplePlan:
    """Resolution contract shared by the router and the future supersampled sampler."""

    version: int
    scale: float
    target_width: int
    target_height: int
    working_width: int
    working_height: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _snap_up(value: float, multiple: int = 8) -> int:
    return int(math.ceil(float(value) / multiple) * multiple)


def build_supersample_plan(
    width: int,
    height: int,
    scale: float,
) -> SupersamplePlan:
    target_width = int(width)
    target_height = int(height)
    requested_scale = max(1.0, min(2.0, float(scale)))

    # Keep the ordinary path exactly the configured size. At larger scales,
    # round upward so "supersampling" can never accidentally remove pixels.
    if requested_scale == 1.0:
        working_width = target_width
        working_height = target_height
    else:
        working_width = _snap_up(target_width * requested_scale)
        working_height = _snap_up(target_height * requested_scale)

    return SupersamplePlan(
        version=1,
        scale=requested_scale,
        target_width=target_width,
        target_height=target_height,
        working_width=working_width,
        working_height=working_height,
    )


def coerce_supersample_plan(value: SupersamplePlan | dict) -> SupersamplePlan:
    if isinstance(value, SupersamplePlan):
        plan = value
    elif isinstance(value, dict):
        try:
            plan = SupersamplePlan(
                version=int(value["version"]),
                scale=float(value["scale"]),
                target_width=int(value["target_width"]),
                target_height=int(value["target_height"]),
                working_width=int(value["working_width"]),
                working_height=int(value["working_height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The supersample plan is missing required resolution values") from exc
    else:
        raise TypeError("supersample_plan must come from Krea2 Multi-LoRA Composer")

    dimensions = (
        plan.target_width,
        plan.target_height,
        plan.working_width,
        plan.working_height,
    )
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("The supersample plan contains a non-positive dimension")
    if plan.working_width < plan.target_width or plan.working_height < plan.target_height:
        raise ValueError("The supersample working canvas cannot be smaller than the target canvas")
    return plan
