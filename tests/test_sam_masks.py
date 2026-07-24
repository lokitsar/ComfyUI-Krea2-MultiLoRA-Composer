import torch

from krea2_router.config import Region
from krea2_router.sam_masks import assign_detections_to_regions, box_iou, normalize_detection_boxes


def _regions():
    return (
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.4, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 1.0, 0.6, 0.0, 0.4, 1.0, 0.0, 1.0),
    )


def test_normalized_sam_boxes_convert_to_pixels():
    boxes = normalize_detection_boxes(torch.tensor([[0.1, 0.2, 0.4, 0.8]]), 1000, 500)
    assert torch.equal(boxes, torch.tensor([[100.0, 100.0, 400.0, 400.0]]))


def test_pairwise_iou_is_one_for_identical_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert float(box_iou(boxes, boxes)[0, 0]) == 1.0


def test_detections_are_reordered_to_character_order():
    detections = torch.tensor([[600.0, 0.0, 1000.0, 500.0], [0.0, 0.0, 400.0, 500.0]])
    assigned, overlaps = assign_detections_to_regions(detections, _regions(), 1000, 500)
    assert assigned == [1, 0]
    assert overlaps == [1.0, 1.0]
