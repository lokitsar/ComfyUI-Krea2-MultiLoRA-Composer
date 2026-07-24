from __future__ import annotations

from typing import Any

KREA2_IM_START_ID = 151644
KREA2_USER_TOKEN_ID = 872
KREA2_NEWLINE_TOKEN_ID = 198


def _collect_token_ids(payload: Any, output: list[int]) -> None:
    if payload is None:
        return
    if isinstance(payload, dict):
        preferred = next(
            (payload[key] for key in ("qwen3vl_4b", "qwen3_4b", "qwen3") if key in payload),
            None,
        )
        if preferred is not None:
            _collect_token_ids(preferred, output)
            return
        for value in payload.values():
            _collect_token_ids(value, output)
        return
    if isinstance(payload, (int, float)):
        output.append(int(payload))
        return
    if isinstance(payload, (tuple, list)):
        is_token_tuple = (
            bool(payload)
            and isinstance(payload[0], (int, float))
            and len(payload) <= 3
            and (len(payload) == 1 or isinstance(payload[1], (int, float)))
        )
        if is_token_tuple:
            output.append(int(payload[0]))
            return
        for item in payload:
            _collect_token_ids(item, output)


def flatten_token_ids(token_payload: Any) -> list[int]:
    output: list[int] = []
    _collect_token_ids(token_payload, output)
    return output


def krea2_template_end(token_ids: list[int]) -> int:
    """Mirror Krea2TEModel's system/user-prefix stripping exactly."""
    template_end = -1
    count_im_start = 0
    for index, token_id in enumerate(token_ids):
        if token_id == KREA2_IM_START_ID and count_im_start < 2:
            template_end = index
            count_im_start += 1
    if template_end < 0:
        return 0
    if (
        len(token_ids) > template_end + 3
        and token_ids[template_end + 1] == KREA2_USER_TOKEN_ID
        and token_ids[template_end + 2] == KREA2_NEWLINE_TOKEN_ID
    ):
        template_end += 3
    return template_end


def _resolve_decoder(clip):
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Krea2 CLIP object has no tokenizer")
    for key in ("qwen3vl_4b", "qwen3_4b", "qwen3"):
        candidate = getattr(tokenizer, key, None)
        if candidate is not None:
            return getattr(candidate, "tokenizer", candidate)
    candidate = getattr(tokenizer, "tokenizer", None)
    if candidate is not None:
        return candidate
    raise ValueError("Could not resolve the Krea2 qwen3vl_4b tokenizer")


def _find_overlapping_tokens(text: str, spans: list[tuple[int, int]], concept: str) -> list[int]:
    if not concept:
        return []
    start = text.find(concept)
    if start < 0:
        start = text.casefold().find(concept.casefold())
    if start < 0:
        return []
    end = start + len(concept)
    return [index for index, (left, right) in enumerate(spans) if right > start and left < end]


def find_krea2_concept_positions(
    clip,
    token_payload: Any,
    concepts: list[str],
    fallbacks: list[str] | None = None,
) -> tuple[list[list[int]], list[str]]:
    """Map exact subject phrases to Krea2's post-template conditioning positions.

    The position convention matches the sequence received by SingleStreamDiT after
    Krea2TEModel removes the fixed system and user-opening prompt prefix.
    """
    token_ids = flatten_token_ids(token_payload)
    if not token_ids:
        raise ValueError("Krea2 tokenizer returned no token IDs")
    decoder = _resolve_decoder(clip)
    stripped_ids = token_ids[krea2_template_end(token_ids) :]
    pieces = [str(decoder.decode([token_id])) for token_id in stripped_ids]
    reconstructed = ""
    spans: list[tuple[int, int]] = []
    for piece in pieces:
        start = len(reconstructed)
        reconstructed += piece
        spans.append((start, len(reconstructed)))

    used_texts: list[str] = []
    positions: list[list[int]] = []
    fallbacks = fallbacks or [""] * len(concepts)
    for index, concept in enumerate(concepts):
        matched = _find_overlapping_tokens(reconstructed, spans, concept)
        used = concept
        if not matched and index < len(fallbacks):
            fallback = fallbacks[index]
            matched = _find_overlapping_tokens(reconstructed, spans, fallback)
            used = fallback
        positions.append(matched)
        used_texts.append(used)
    return positions, used_texts
