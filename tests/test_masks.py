import json

import torch

from krea2_router.config import parse_regions
from krea2_router.masks import (
    build_region_masks,
    build_routing_masks,
    masks_to_preview,
    resize_mask_batch,
    resolve_mask_overlaps,
)


def _regions():
    return parse_regions(json.dumps([
        {"name": "A", "lora": "a", "trigger": "A", "x": .1, "y": .1, "w": .55, "h": .8},
        {"name": "B", "lora": "b", "trigger": "B", "x": .35, "y": .1, "w": .55, "h": .8},
    ]))[0]


def test_inward_feather_is_exactly_zero_outside_boxes():
    masks = build_region_masks(64, 64, _regions(), feather=.15, overlap_policy="allow")
    assert torch.count_nonzero(masks[0, :, :6]) == 0
    assert torch.count_nonzero(masks[1, :, 58:]) == 0


def test_nearest_overlap_never_activates_two_regions_at_once():
    masks = build_region_masks(64, 64, _regions(), feather=.05, overlap_policy="nearest")
    assert int((masks > 0).sum(dim=0).max()) == 1


def test_preview_is_comfy_image_shape():
    preview = masks_to_preview(build_region_masks(32, 48, _regions(), .08, "nearest"))
    assert preview.shape == (1, 32, 48, 3)
    assert 0 <= float(preview.min()) <= float(preview.max()) <= 1


def test_external_masks_resize_and_resolve_overlap():
    regions = _regions()
    masks = torch.ones((2, 8, 8))
    resized = resize_mask_batch(masks, 4, 6)
    resolved = resolve_mask_overlaps(resized, regions, "nearest")
    assert resolved.shape == (2, 4, 6)
    assert int((resolved > 0).sum(dim=0).max()) == 1


def test_unboxed_canvas_mask_is_inverse_of_all_enabled_character_boxes():
    characters = _regions()
    canvas = parse_regions(json.dumps([
        {"name": "Canvas LoRA", "lora": "style", "x": 0, "y": 0, "w": 1, "h": 1},
    ]))[0][0]
    routed = build_routing_masks(
        32,
        32,
        [characters[0], canvas],
        feather=0.0,
        mask_modes=["region", "unboxed"],
        exclusion_regions=characters,
    )
    exclusions = build_region_masks(32, 32, characters, feather=0.0, overlap_policy="allow")
    assert torch.equal(routed[1], 1.0 - exclusions.amax(dim=0))
    assert torch.count_nonzero(routed[1] * exclusions.amax(dim=0)) == 0


def test_global_canvas_mask_covers_character_boxes_too():
    character = _regions()[0]
    canvas = parse_regions(json.dumps([
        {"name": "Canvas LoRA", "lora": "style", "x": 0, "y": 0, "w": 1, "h": 1},
    ]))[0][0]
    routed = build_routing_masks(
        8,
        8,
        [character, canvas],
        feather=0.0,
        mask_modes=["region", "global"],
        exclusion_regions=[character],
    )
    assert torch.all(routed[1] == 1.0)


def test_unboxed_canvas_and_single_character_exchange_strength_across_feather():
    character = _regions()[0]
    canvas = parse_regions(json.dumps([
        {"name": "Canvas LoRA", "lora": "style", "x": 0, "y": 0, "w": 1, "h": 1},
    ]))[0][0]
    routed = build_routing_masks(
        32,
        32,
        [character, canvas],
        feather=0.2,
        mask_modes=["region", "unboxed"],
        exclusion_regions=[character],
    )
    assert torch.allclose(routed[0] + routed[1], torch.ones((32, 32)))
