from krea2_router.tokens import find_krea2_concept_positions, flatten_token_ids, krea2_template_end


class _Decoder:
    pieces = {
        10: "Character Alpha",
        11: " and ",
        12: "Character Beta",
        13: ".",
    }

    def decode(self, token_ids):
        return self.pieces.get(token_ids[0], "")


class _TokenizerBranch:
    tokenizer = _Decoder()


class _Tokenizer:
    qwen3vl_4b = _TokenizerBranch()


class _Clip:
    tokenizer = _Tokenizer()


def _payload():
    ids = [151644, 99, 151644, 872, 198, 10, 11, 12, 13]
    return {"qwen3vl_4b": [[(token_id, 1.0) for token_id in ids]]}


def test_krea2_positions_align_after_stripped_prompt_prefix():
    positions, used = find_krea2_concept_positions(
        _Clip(),
        _payload(),
        ["Character Alpha", "Character Beta"],
    )
    assert positions == [[0], [2]]
    assert used == ["Character Alpha", "Character Beta"]


def test_concept_position_falls_back_to_trigger():
    positions, used = find_krea2_concept_positions(
        _Clip(),
        _payload(),
        ["missing long description"],
        ["Character Beta"],
    )
    assert positions == [[2]]
    assert used == ["Character Beta"]


def test_token_payload_flattening_and_template_offset_match_krea2():
    ids = flatten_token_ids(_payload())
    assert ids[-4:] == [10, 11, 12, 13]
    assert krea2_template_end(ids) == 5
