from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


DEFAULT_REGIONS_JSON = json.dumps(
    [
        {
            "name": "Character A",
            "enabled": True,
            "lora": "None",
            "trigger": "CHAR_A",
            "prompt": "",
            "strength": 1.0,
            "x": 0.05,
            "y": 0.08,
            "w": 0.40,
            "h": 0.84,
            "start": 0.0,
            "end": 1.0,
        },
        {
            "name": "Character B",
            "enabled": True,
            "lora": "None",
            "trigger": "CHAR_B",
            "prompt": "",
            "strength": 1.0,
            "x": 0.55,
            "y": 0.08,
            "w": 0.40,
            "h": 0.84,
            "start": 0.0,
            "end": 1.0,
        },
    ],
    indent=2,
)


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


@dataclass(frozen=True)
class Region:
    name: str
    enabled: bool
    lora: str
    trigger: str
    prompt: str
    strength: float
    x: float
    y: float
    w: float
    h: float
    start: float
    end: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass(frozen=True)
class RouterConfig:
    feather: float = 0.08
    overlap_policy: str = "nearest"
    schedule_softness: float = 0.04
    attention_bias: bool = True
    negative_bias: float = 5.0
    positive_bias: float = 1.0
    bias_block_fraction: float = 0.0
    strict: bool = True
    debug: bool = False


def _region_from_item(item: dict[str, Any], index: int) -> Region:
    name = _clean_text(item.get("name")) or f"Character {index + 1}"
    lora = _clean_text(item.get("lora", item.get("lora_name", "None"))) or "None"
    trigger = _clean_text(item.get("trigger"))
    prompt = _clean_text(item.get("prompt", item.get("description", "")))
    strength = _finite_float(item.get("strength", 1.0), 1.0)

    if "x1" in item or "y1" in item:
        x = _finite_float(item.get("x", item.get("x0", 0.0)), 0.0)
        y = _finite_float(item.get("y", item.get("y0", 0.0)), 0.0)
        x1 = _finite_float(item.get("x1", 1.0), 1.0)
        y1 = _finite_float(item.get("y1", 1.0), 1.0)
        w, h = x1 - x, y1 - y
    else:
        x = _finite_float(item.get("x", 0.0), 0.0)
        y = _finite_float(item.get("y", 0.0), 0.0)
        w = _finite_float(item.get("w", item.get("width", 1.0)), 1.0)
        h = _finite_float(item.get("h", item.get("height", 1.0)), 1.0)

    x = _clamp(x, 0.0, 1.0)
    y = _clamp(y, 0.0, 1.0)
    w = _clamp(w, 0.001, 1.0 - x)
    h = _clamp(h, 0.001, 1.0 - y)
    start = _clamp(_finite_float(item.get("start", 0.0), 0.0), 0.0, 1.0)
    end = _clamp(_finite_float(item.get("end", 1.0), 1.0), 0.0, 1.0)

    return Region(
        name=name,
        enabled=bool(item.get("enabled", item.get("enable", True))),
        lora=lora,
        trigger=trigger,
        prompt=prompt,
        strength=strength,
        x=x,
        y=y,
        w=w,
        h=h,
        start=start,
        end=end,
    )


def parse_regions(regions_json: str) -> tuple[list[Region], list[str]]:
    """Parse the stable JSON schema and return regions plus actionable warnings."""
    try:
        raw = json.loads(regions_json or "[]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"regions_json is not valid JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("regions_json must be a JSON array")

    regions = [_region_from_item(item, index) for index, item in enumerate(raw) if isinstance(item, dict)]
    warnings: list[str] = []
    seen_names: set[str] = set()
    seen_triggers: set[str] = set()
    for region in regions:
        folded_name = region.name.casefold()
        if folded_name in seen_names:
            warnings.append(f"Duplicate region name: {region.name}")
        seen_names.add(folded_name)

        folded_trigger = region.trigger.casefold()
        if region.enabled and not region.trigger:
            warnings.append(f"{region.name} has no trigger phrase")
        elif folded_trigger and folded_trigger in seen_triggers:
            warnings.append(f"Duplicate trigger phrase: {region.trigger}")
        if folded_trigger:
            seen_triggers.add(folded_trigger)

        if region.enabled and region.lora in {"", "None"}:
            warnings.append(f"{region.name} has no LoRA selected")
        if region.start >= region.end:
            warnings.append(f"{region.name} has an empty denoising schedule ({region.start:.2f}-{region.end:.2f})")
        if abs(region.strength) > 2.0:
            warnings.append(f"{region.name} uses a high LoRA strength ({region.strength:.2f})")

    for left_index, left in enumerate(regions):
        if not left.enabled:
            continue
        lx0, ly0, lx1, ly1 = left.box
        for right in regions[left_index + 1 :]:
            if not right.enabled:
                continue
            rx0, ry0, rx1, ry1 = right.box
            intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
                0.0, min(ly1, ry1) - max(ly0, ry0)
            )
            smaller_area = min(left.w * left.h, right.w * right.h)
            if smaller_area > 0 and intersection / smaller_area > 0.10:
                warnings.append(
                    f"{left.name} and {right.name} overlap by {intersection / smaller_area:.0%}; "
                    "nearest-region arbitration will split the overlap"
                )
    return regions, warnings


def _location_phrase(region: Region) -> str:
    cx, cy = region.center
    horizontal = "left" if cx < 0.40 else "right" if cx > 0.60 else "center"
    vertical = "upper" if cy < 0.35 else "lower" if cy > 0.65 else ""
    return f"in the {vertical + ' ' if vertical else ''}{horizontal} of the image"


def concept_text_for_region(region: Region) -> str:
    """Return the exact subject phrase embedded in the composed prompt."""
    character = region.prompt
    if region.trigger and region.trigger.casefold() not in character.casefold():
        character = f"{region.trigger}, {character}" if character else region.trigger
    return character


def compose_prompt(scene_prompt: str, regions: list[Region]) -> str:
    parts = [_clean_text(scene_prompt)]
    for region in regions:
        if not region.enabled:
            continue
        character = concept_text_for_region(region)
        if character:
            parts.append(f"{_location_phrase(region)}, {character}")
    return ". ".join(part.rstrip(" .") for part in parts if part).strip() + "."
