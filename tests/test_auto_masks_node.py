import importlib.util
import sys
from pathlib import Path

import torch

from krea2_router.config import Region, RouterConfig


def _load_nodes_module():
    root = Path(__file__).resolve().parents[1]
    package = "k2cr_auto_masks_test"
    spec = importlib.util.spec_from_file_location(
        package,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return sys.modules[f"{package}.nodes"]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class _Processor:
    def __init__(self):
        self.device = "cpu"
        self.threshold = None

    def set_confidence_threshold(self, value):
        self.threshold = value

    def set_image(self, _image):
        return {}

    def add_multiple_box_prompts(self, boxes, labels, state):
        assert len(boxes) == 2
        assert labels == [True, True]
        state["masks"] = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        )
        # Deliberately return B before A so the node must reorder the masks.
        state["boxes"] = torch.tensor([[1.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0]])
        state["scores"] = torch.tensor([0.9, 0.8])
        return state


def _plan(nodes):
    regions = (
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0),
    )
    return nodes.RoutePlan(None, "", regions, (), RouterConfig(), ((0,), (1,)), ("A", "B"), (), 2, 2)


def test_auto_masks_node_executes_and_orders_masks():
    nodes = _load_nodes_module()
    model = _Model()
    sam3_model = {"model": model, "processor": _Processor(), "device": "cpu", "original_device": "cpu"}
    image = torch.zeros((1, 2, 2, 3))
    masks, preview, diagnostics = nodes.Krea2SAM3AutoMasks().segment(sam3_model, image, _plan(nodes), 0.2)
    assert masks.shape == (2, 2, 2)
    assert float(masks[0, 0, 0]) == 1.0
    assert float(masks[1, 1, 1]) == 1.0
    assert preview.shape == (1, 2, 2, 3)
    assert '"selected_indices": [' in diagnostics
