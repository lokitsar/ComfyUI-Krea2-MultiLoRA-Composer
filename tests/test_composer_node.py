import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def composer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = "k2cr_composer_test"
    spec = importlib.util.spec_from_file_location(
        package, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    spec.loader.exec_module(module)
    node_module = sys.modules[f"{package}.nodes"]
    management = ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    management.intermediate_dtype = lambda: torch.float32
    comfy = ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    sessions = []
    monkeypatch.setattr(node_module, "install_router_wrapper", lambda model, session: sessions.append(session))
    monkeypatch.setattr(
        node_module, "load_region_adapters",
        lambda model, region, strict: SimpleNamespace(adapters={}, skipped_keys=[], unsupported=[]),
    )
    monkeypatch.setattr(
        node_module, "find_krea2_concept_positions",
        lambda clip, tokens, concepts, fallbacks: (
            [[index] if text else [] for index, text in enumerate(concepts)], concepts,
        ),
    )
    model = SimpleNamespace(clone=lambda: object())
    clip = SimpleNamespace(tokenize=lambda prompt: prompt, encode_from_tokens_scheduled=lambda tokens: [])

    def apply(canvases, characters=None):
        return node_module.Krea2CharacterRouter().apply(
            model=model, clip=clip, scene_prompt="A city", width=256, height=256,
            regions_json=json.dumps(
                [{"lora": "person", "trigger": "PERSON", "x": 0, "w": 0.5}]
                if characters is None else characters
            ),
            feather=0.0, overlap_policy="nearest", schedule_softness=0.0,
            strict=True, debug=False, supersample_scale=1.0,
            canvas_lora_json=json.dumps(canvases),
        )

    return apply, sessions, node_module


@pytest.mark.parametrize("canvases", [
    {"enabled": True, "lora": "legacy", "coverage": "global"},
    [
        {"enabled": True, "lora": "paint", "trigger": "PAINT", "coverage": "global", "start": 0.2},
        {"enabled": False, "lora": "disabled", "trigger": "DISABLED"},
        {"enabled": True, "lora": "zero", "trigger": "ZERO", "strength": 0},
        {"enabled": True, "lora": "grain", "prompt": "paper grain", "coverage": "unboxed", "end": 0.8},
        {"enabled": True, "lora": "lighting", "coverage": "global", "strength": -0.5},
    ],
    [{"enabled": True, "lora": f"style-{index}", "coverage": "global"} for index in range(8)],
    [],
])
def test_composer_routes_and_exports_all_canvas_loras(composer, canvases):
    apply, sessions, node_module = composer
    output = apply(canvases)
    parsed, _ = node_module.parse_canvas_loras(json.dumps(canvases))
    active = [(index, canvas) for index, canvas in enumerate(parsed)
              if canvas.enabled and canvas.lora != "None" and canvas.strength != 0]
    session = sessions[0]
    assert [region.lora for region in session.regions] == ["person", *[canvas.lora for _, canvas in active]]
    assert session.mask_modes == ["region", *[canvas.coverage for _, canvas in active]]
    assert len(session.loaded) == len(session.token_positions) == 1 + len(active)
    assert [(region.start, region.end) for region in session.regions[1:]] == [
        (canvas.start, canvas.end) for _, canvas in active
    ]
    assert len({region.name for region in session.regions}) == len(session.regions)
    diagnostics = json.loads(output[4])
    assert [entry["mask_mode"] for entry in diagnostics["regions"]] == session.mask_modes
    assert "DISABLED" not in output[2]
    assert "ZERO" not in output[2]
    shared = json.loads(output[6])
    restored, _ = node_module.parse_canvas_loras(json.dumps(shared["canvas_loras"]))
    assert restored == parsed
    assert shared["format"] == "krea2_character_router_share_v1"
    assert shared["canvas_lora"] == (shared["canvas_loras"][0] if parsed else {"enabled": False})


def test_composer_supports_multiple_canvas_loras_without_active_characters(composer):
    apply, sessions, _ = composer
    output = apply([
        {"enabled": True, "lora": "paint", "coverage": "global"},
        {"enabled": True, "lora": "texture", "coverage": "unboxed"},
    ], characters=[])
    assert sessions[0].mask_modes == ["global", "unboxed"]
    assert sessions[0].token_positions == [[], []]
    assert output[2] == "A city."
