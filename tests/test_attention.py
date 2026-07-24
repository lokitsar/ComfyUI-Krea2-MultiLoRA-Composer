import torch

from krea2_router.attention import construct_box_attention_bias


def test_box_attention_bias_favors_own_text_and_suppresses_competitor():
    masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).reshape(2, 1, 2)
    bias = construct_box_attention_bias(
        masks,
        token_positions=[[0], [1]],
        text_tokens=2,
        negative_bias=5.0,
        positive_bias=1.0,
    )
    assert bias.shape == (1, 4, 4)
    # Image token A -> own text A / competing text B.
    assert float(bias[0, 2, 0]) == 1.0
    assert float(bias[0, 2, 1]) == -5.0
    # Image token B -> competing text A / own text B.
    assert float(bias[0, 3, 0]) == -5.0
    assert float(bias[0, 3, 1]) == 1.0
    # Subject text is discouraged from attending outside its assigned box.
    assert float(bias[0, 0, 2]) == 1.0
    assert float(bias[0, 0, 3]) == -5.0


def test_box_attention_bias_leaves_shared_text_and_image_to_image_neutral():
    masks = torch.tensor([[1.0, 0.0]])
    bias = construct_box_attention_bias(masks, [[0]], text_tokens=2)
    assert torch.count_nonzero(bias[0, :, 1]) == 0
    assert torch.count_nonzero(bias[0, 2:, 2:]) == 0


def test_unboxed_gap_suppresses_every_subject_phrase():
    masks = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    bias = construct_box_attention_bias(
        masks,
        token_positions=[[0], [1]],
        text_tokens=2,
        negative_bias=5.0,
        positive_bias=1.0,
    )
    # The middle image token is outside both character boxes.
    assert float(bias[0, 3, 0]) == -5.0
    assert float(bias[0, 3, 1]) == -5.0
    # Each boxed image token still favors only its own subject.
    assert float(bias[0, 2, 0]) == 1.0
    assert float(bias[0, 2, 1]) == -5.0
    assert float(bias[0, 4, 0]) == -5.0
    assert float(bias[0, 4, 1]) == 1.0
