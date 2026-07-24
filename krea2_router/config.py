from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
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

DEFAULT_CANVAS_LORA_JSON = json.dumps(
    {
        "enabled": False,
        "lora": "None",
        "trigger": "",
        "prompt": "",
        "strength": 1.0,
        "coverage": "unboxed",
        "start": 0.0,
        "end": 1.0,
    },
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
class CanvasLora:
    enabled: bool
    lora: str
    trigger: str
    prompt: str
    strength: float
    coverage: str
    start: float
    end: float

    def as_region(self) -> Region:
        return Region(
            name="Canvas LoRA",
            enabled=self.enabled,
            lora=self.lora,
            trigger=self.trigger,
            prompt=self.prompt,
            strength=self.strength,
            x=0.0,
            y=0.0,
            w=1.0,
            h=1.0,
            start=self.start,
            end=self.end,
        )


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


def parse_canvas_lora(canvas_lora_json: str) -> tuple[CanvasLora, list[str]]:
    """Parse the optional full-canvas or inverse-character LoRA configuration."""
    try:
        raw = json.loads(canvas_lora_json or DEFAULT_CANVAS_LORA_JSON)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canvas_lora_json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("canvas_lora_json must be a JSON object")

    coverage = _clean_text(raw.get("coverage", "unboxed")).casefold()
    warnings: list[str] = []
    if coverage not in {"unboxed", "global"}:
        warnings.append(f"Canvas LoRA has unknown coverage mode {coverage!r}; using unboxed")
        coverage = "unboxed"

    canvas = CanvasLora(
        enabled=bool(raw.get("enabled", False)),
        lora=_clean_text(raw.get("lora", raw.get("lora_name", "None"))) or "None",
        trigger=_clean_text(raw.get("trigger")),
        prompt=_clean_text(raw.get("prompt", raw.get("description", ""))),
        strength=_finite_float(raw.get("strength", 1.0), 1.0),
        coverage=coverage,
        start=_clamp(_finite_float(raw.get("start", 0.0), 0.0), 0.0, 1.0),
        end=_clamp(_finite_float(raw.get("end", 1.0), 1.0), 0.0, 1.0),
    )
    if canvas.enabled:
        if canvas.lora in {"", "None"}:
            warnings.append("Canvas LoRA is enabled but no LoRA is selected")
        if canvas.start >= canvas.end:
            warnings.append(
                f"Canvas LoRA has an empty denoising schedule ({canvas.start:.2f}-{canvas.end:.2f})"
            )
        if abs(canvas.strength) > 2.0:
            warnings.append(f"Canvas LoRA uses a high strength ({canvas.strength:.2f})")
    return canvas, warnings


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


def concept_text_for_canvas(canvas_lora: CanvasLora) -> str:
    """Return the optional Canvas LoRA phrase without inventing a spatial subject."""
    concept = canvas_lora.prompt
    if canvas_lora.trigger and canvas_lora.trigger.casefold() not in concept.casefold():
        concept = f"{canvas_lora.trigger}, {concept}" if concept else canvas_lora.trigger
    return concept


def compose_prompt(
    scene_prompt: str,
    regions: list[Region],
    canvas_lora: CanvasLora | None = None,
) -> str:
    parts = [_clean_text(scene_prompt)]
    if canvas_lora is not None and canvas_lora.enabled:
        canvas_concept = concept_text_for_canvas(canvas_lora)
        if canvas_concept:
            parts.append(canvas_concept)
    for region in regions:
        if not region.enabled:
            continue
        character = concept_text_for_region(region)
        if character:
            parts.append(f"{_location_phrase(region)}, {character}")
    return ". ".join(part.rstrip(" .") for part in parts if part).strip() + "."
