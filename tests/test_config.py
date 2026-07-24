import json

from krea2_router.config import (
    DEFAULT_CANVAS_LORA_JSON,
    compose_prompt,
    concept_text_for_canvas,
    concept_text_for_region,
    parse_canvas_lora,
    parse_regions,
)


def test_parse_regions_clamps_coordinates_and_accepts_description_alias():
    regions, warnings = parse_regions(json.dumps([{
        "name": "A", "lora": "a.safetensors", "trigger": "ALPHA",
        "description": "red coat", "x": -1, "y": 0.2, "w": 2, "h": 0.5,
    }]))
    assert not warnings
    assert regions[0].box == (0.0, 0.2, 1.0, 0.7)
    assert regions[0].prompt == "red coat"


def test_compose_prompt_places_trigger_exactly_once():
    regions, _ = parse_regions(json.dumps([{
        "name": "A", "lora": "a.safetensors", "trigger": "ALPHA",
        "prompt": "ALPHA wearing a red coat", "x": 0.0, "y": 0.0, "w": 0.3, "h": 1.0,
    }]))
    prompt = compose_prompt("A rainy city", regions)
    assert prompt.count("ALPHA") == 1
    assert "left of the image" in prompt
    assert prompt.endswith(".")
    assert concept_text_for_region(regions[0]) == "ALPHA wearing a red coat"


def test_duplicate_trigger_and_overlap_are_reported():
    regions, warnings = parse_regions(json.dumps([
        {"name": "A", "lora": "a.safetensors", "trigger": "same", "x": 0, "y": 0, "w": .7, "h": 1},
        {"name": "B", "lora": "b.safetensors", "trigger": "same", "x": .3, "y": 0, "w": .7, "h": 1},
    ]))
    assert len(regions) == 2
    assert any("Duplicate trigger" in warning for warning in warnings)
    assert any("overlap" in warning for warning in warnings)


def test_canvas_lora_defaults_disabled_for_existing_workflows():
    canvas, warnings = parse_canvas_lora(DEFAULT_CANVAS_LORA_JSON)
    assert not warnings
    assert canvas.enabled is False
    assert canvas.coverage == "unboxed"
    assert canvas.lora == "None"


def test_canvas_lora_phrase_is_composed_before_character_regions():
    regions, _ = parse_regions(json.dumps([{
        "name": "A", "lora": "a.safetensors", "trigger": "ALPHA",
        "prompt": "wearing a red coat", "x": 0.0, "y": 0.0, "w": 0.3, "h": 1.0,
    }]))
    canvas, warnings = parse_canvas_lora(json.dumps({
        "enabled": True,
        "lora": "style.safetensors",
        "trigger": "STYLE_TOKEN",
        "description": "painted city environment",
        "coverage": "unboxed",
    }))
    assert not warnings
    prompt = compose_prompt("A rainy city", regions, canvas)
    assert prompt.index("STYLE_TOKEN") < prompt.index("left of the image")
    assert prompt.count("STYLE_TOKEN") == 1
    assert concept_text_for_canvas(canvas) == "STYLE_TOKEN, painted city environment"


def test_canvas_lora_allows_triggerless_style_adapters():
    canvas, warnings = parse_canvas_lora(json.dumps({
        "enabled": True,
        "lora": "style.safetensors",
        "description": "hand-painted scenery",
        "coverage": "global",
    }))
    assert not warnings
    assert concept_text_for_canvas(canvas) == "hand-painted scenery"
    assert canvas.as_region().box == (0.0, 0.0, 1.0, 1.0)
