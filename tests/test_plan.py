import pytest

from krea2_router.config import Region, RouterConfig
from krea2_router.plan import RoutePlan, sam3_box_prompt


def _plan():
    region = Region("A", True, "a", "A", "", 1.0, 0.1, 0.2, 0.3, 0.5, 0.0, 1.0)
    return RoutePlan(None, "", (region,), (), RouterConfig(), ((0,),), ("A",), (), 1000, 800)


def test_sam3_box_prompt_uses_normalized_center_format():
    prompt, region = sam3_box_prompt(_plan(), 1, padding=0.0)
    assert region.name == "A"
    assert prompt["labels"] == [True]
    assert prompt["boxes"][0] == pytest.approx([0.25, 0.45, 0.3, 0.5])


def test_sam3_box_prompt_rejects_missing_character():
    try:
        sam3_box_prompt(_plan(), 2)
    except ValueError as exc:
        assert "route plan has 1 characters" in str(exc)
    else:
        raise AssertionError("missing character should fail")
