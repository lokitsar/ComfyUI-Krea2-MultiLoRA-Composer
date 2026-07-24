import json

import torch

from krea2_router.config import parse_regions
from krea2_router.masks import (
    build_region_masks,
    masks_to_preview,
    prepare_external_masks,
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


def test_external_mask_batch_count_must_match_characters():
    masks = torch.zeros((1, 16, 16))
    try:
        prepare_external_masks(masks, expected=2)
    except ValueError as exc:
        assert "Character A, B, C order" in str(exc)
    else:
        raise AssertionError("mask count mismatch should fail")


def test_external_mask_grow_and_feather_create_soft_foreground_boundary():
    masks = torch.zeros((1, 17, 17))
    masks[0, 8, 8] = 1
    prepared = prepare_external_masks(masks, expected=1, grow=2, feather=2)
    assert prepared.shape == masks.shape
    assert float(prepared[0, 8, 8]) > 0.9
    assert 0.0 < float(prepared[0, 8, 5]) < 1.0
    assert float(prepared[0, 0, 0]) == 0.0


def test_external_masks_resize_and_resolve_overlap():
    regions = _regions()
    masks = torch.ones((2, 8, 8))
    resized = resize_mask_batch(masks, 4, 6)
    resolved = resolve_mask_overlaps(resized, regions, "nearest")
    assert resolved.shape == (2, 4, 6)
    assert int((resolved > 0).sum(dim=0).max()) == 1
