import pytest

from krea2_router.supersampling import build_supersample_plan, coerce_supersample_plan


def test_one_x_preserves_the_exact_canvas():
    plan = build_supersample_plan(1352, 768, 1.0)
    assert plan.target_width == 1352
    assert plan.target_height == 768
    assert plan.working_width == 1352
    assert plan.working_height == 768


def test_supersampling_uses_a_larger_multiple_of_eight_canvas():
    plan = build_supersample_plan(1216, 832, 1.25)
    assert plan.working_width == 1520
    assert plan.working_height == 1040
    assert plan.as_dict() == {
        "version": 1,
        "scale": 1.25,
        "target_width": 1216,
        "target_height": 832,
        "working_width": 1520,
        "working_height": 1040,
    }


def test_non_integral_scaled_dimensions_round_up_without_changing_aspect_materially():
    plan = build_supersample_plan(1352, 768, 1.1)
    assert plan.working_width == 1488
    assert plan.working_height == 848
    assert plan.working_width >= 1352 * 1.1
    assert plan.working_height >= 768 * 1.1


def test_scale_is_clamped_to_supported_range():
    assert build_supersample_plan(1024, 1024, 0.5).scale == 1.0
    assert build_supersample_plan(1024, 1024, 3.0).scale == 2.0


def test_plan_can_be_reconstructed_from_a_runtime_dictionary():
    source = build_supersample_plan(1216, 832, 1.25)
    assert coerce_supersample_plan(source.as_dict()) == source


def test_plan_rejects_a_working_canvas_smaller_than_its_target():
    with pytest.raises(ValueError, match="cannot be smaller"):
        coerce_supersample_plan(
            {
                "version": 1,
                "scale": 1.25,
                "target_width": 1216,
                "target_height": 832,
                "working_width": 1024,
                "working_height": 768,
            }
        )
