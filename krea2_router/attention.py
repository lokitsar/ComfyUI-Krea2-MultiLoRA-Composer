from __future__ import annotations

import torch


def construct_box_attention_bias(
    region_masks: torch.Tensor,
    token_positions: list[list[int]],
    text_tokens: int,
    negative_bias: float = 5.0,
    positive_bias: float = 1.0,
    bidirectional: bool = True,
) -> torch.Tensor | None:
    """Build FreeFuse-style subject/text bias from explicit region masks.

    Layout is Krea2's joint ``[text, image]`` sequence. Image tokens inside a
    character region favor that character's phrase and suppress other character
    phrases. In the reverse direction, character text favors its assigned box.
    """
    if region_masks.numel() == 0 or not any(token_positions):
        return None
    masks = region_masks.reshape(region_masks.shape[0], -1)
    image_tokens = masks.shape[1]
    total = text_tokens + image_tokens
    bias = torch.zeros((1, total, total), device=masks.device, dtype=masks.dtype)

    valid_positions = [
        sorted({position for position in positions if 0 <= position < text_tokens})
        for positions in token_positions
    ]
    for region_index, image_mask in enumerate(masks):
        own = valid_positions[region_index] if region_index < len(valid_positions) else []
        if not own:
            continue
        # Treat the drawn box as the complete allowed area for this subject.
        # Merely suppressing "other subjects inside this box" leaves unboxed gaps
        # neutral, allowing a character to drift into the center. Penalizing the
        # subject's own phrase everywhere outside its mask closes that loophole.
        outside = 1.0 - image_mask
        bias[0, text_tokens:, own] -= outside[:, None] * float(negative_bias)
        if positive_bias:
            bias[0, text_tokens:, own] += image_mask[:, None] * float(positive_bias)
        if bidirectional:
            bias[0, own, text_tokens:] -= outside[None, :] * float(negative_bias)
            if positive_bias:
                bias[0, own, text_tokens:] += image_mask[None, :] * float(positive_bias)
    return bias
