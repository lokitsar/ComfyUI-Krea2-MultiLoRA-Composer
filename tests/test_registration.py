import importlib.util
import sys
from pathlib import Path


def test_router_and_supersampled_sampler_are_registered():
    root = Path(__file__).resolve().parents[1]
    package = "k2cr_registration_test"
    spec = importlib.util.spec_from_file_location(
        package,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    assert module.NODE_CLASS_MAPPINGS.keys() == {
        "Krea2CharacterRouter",
        "Krea2SupersampledKSampler",
    }
    assert module.NODE_CLASS_MAPPINGS["Krea2CharacterRouter"].RETURN_TYPES == (
        "MODEL",
        "CONDITIONING",
        "STRING",
        "IMAGE",
        "STRING",
        "LATENT",
        "STRING",
        "INT",
        "INT",
        "KREA2_SUPERSAMPLE_PLAN",
    )
    assert list(module.NODE_CLASS_MAPPINGS["Krea2CharacterRouter"].INPUT_TYPES()["required"])[-1] == (
        "supersample_scale"
    )
    sampler = module.NODE_CLASS_MAPPINGS["Krea2SupersampledKSampler"]
    assert sampler.RETURN_TYPES == ("IMAGE", "LATENT", "IMAGE", "STRING")
    assert sampler.RETURN_NAMES == ("image", "latent", "working_image", "diagnostics")
