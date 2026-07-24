import torch

from krea2_router.adapters import module_path_from_target
from krea2_router.schedule import sampling_progress, schedule_weight


def test_only_spatial_krea_modules_are_routeable():
    assert module_path_from_target("diffusion_model.blocks.3.attn.wq.weight") == "blocks.3.attn.wq"
    assert module_path_from_target("diffusion_model.first.weight") == "first"
    assert module_path_from_target("diffusion_model.last.linear.weight") == "last.linear"
    assert (
        module_path_from_target("diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight")
        == "txtfusion.refiner_blocks.0.attn.wq"
    )
    assert module_path_from_target("diffusion_model.txtmlp.1.weight") is None


def test_schedule_has_soft_edges_and_zero_outside():
    assert schedule_weight(0, 0, .8, .05) == 1
    assert schedule_weight(.1, .2, .8, .05) == 0
    assert schedule_weight(.2, .2, .8, .05) == 0
    assert 0 < schedule_weight(.225, .2, .8, .05) < 1
    assert schedule_weight(.5, .2, .8, .05) == 1
    assert schedule_weight(.81, .2, .8, .05) == 0


def test_sampling_progress_uses_nearest_sigma():
    options = {"sigmas": torch.tensor([.5]), "sample_sigmas": torch.tensor([1., .75, .5, .25, 0.])}
    assert sampling_progress(options) == .5
