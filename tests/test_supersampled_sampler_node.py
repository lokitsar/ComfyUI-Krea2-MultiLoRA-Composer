import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_custom_node_package():
    root = Path(__file__).resolve().parents[1]
    package = "k2cr_supersampled_sampler_test"
    spec = importlib.util.spec_from_file_location(
        package,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return module, sys.modules[f"{package}.nodes"]


class _FakeVAE:
    def decode(self, latent):
        batch = int(latent.shape[0])
        return torch.ones((batch, 32, 48, 3), dtype=torch.float32)


def test_sampler_decodes_working_image_and_returns_original_target_size(monkeypatch):
    module, node_module = _load_custom_node_package()
    sampler = module.NODE_CLASS_MAPPINGS["Krea2SupersampledKSampler"]()
    latent = {
        "samples": torch.zeros((1, 4, 4, 6), dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }

    def fake_common_ksampler(*args, **kwargs):
        output = latent.copy()
        output["samples"] = latent["samples"].clone()
        return output

    def fake_decode_and_downscale(*args, **kwargs):
        working = torch.ones((1, 32, 48, 3), dtype=torch.float32)
        final = torch.ones((1, 16, 24, 3), dtype=torch.float32)
        return final, working

    monkeypatch.setattr(node_module, "_run_common_ksampler", fake_common_ksampler)
    monkeypatch.setattr(node_module, "_decode_and_downscale", fake_decode_and_downscale)
    final, sampled, working, diagnostics = sampler.sample(
        model=object(),
        positive=[],
        negative=[],
        latent_image=latent,
        vae=_FakeVAE(),
        supersample_plan={
            "version": 1,
            "scale": 2.0,
            "target_width": 24,
            "target_height": 16,
            "working_width": 48,
            "working_height": 32,
        },
        seed=123,
        steps=8,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
        downscale_method="area",
    )

    assert tuple(final.shape) == (1, 16, 24, 3)
    assert tuple(working.shape) == (1, 32, 48, 3)
    assert sampled["samples"].shape == latent["samples"].shape
    assert json.loads(diagnostics)["target"] == [24, 16]


def test_sampler_rejects_latent_from_a_different_router_plan():
    module, _ = _load_custom_node_package()
    sampler = module.NODE_CLASS_MAPPINGS["Krea2SupersampledKSampler"]()
    latent = {
        "samples": torch.zeros((1, 4, 4, 6), dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }

    try:
        sampler.sample(
            model=object(),
            positive=[],
            negative=[],
            latent_image=latent,
            vae=_FakeVAE(),
            supersample_plan={
                "version": 1,
                "scale": 2.0,
                "target_width": 32,
                "target_height": 32,
                "working_width": 64,
                "working_height": 64,
            },
            seed=0,
            steps=8,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
            downscale_method="area",
        )
    except ValueError as error:
        assert "Connect latent and supersample_plan from the same" in str(error)
    else:
        raise AssertionError("Expected a mismatched latent to be rejected")
