import torch

from krea2_router.adapters import LoadedRegionAdapters
from krea2_router.config import Region, RouterConfig
from krea2_router.runtime import RouterSession


class _FakeAdapter:
    def h(self, source, _base_output):
        return torch.ones((*source.shape[:-1], 1), device=source.device, dtype=source.dtype) * self.multiplier

    def g(self, value):
        return value


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.proj.weight)


class _FakeKrea(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.txtfusion = torch.nn.Identity()
        self.txtmlp = torch.nn.Identity()
        self.patch = 2

    def _unpack_context(self, value):
        return value


class _Executor:
    def __init__(self, model):
        self.class_obj = model
        self.model_args = None
        self.model_kwargs = None

    def __call__(self, x, _timesteps, context, *model_args, **model_kwargs):
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        sequence = torch.zeros((x.shape[0], context.shape[1] + 4, 1), dtype=x.dtype, device=x.device)
        return self.class_obj.blocks[0].proj(sequence)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_mask = None

    def forward(self, value, _freqs=None, mask=None, transformer_options=None):
        self.received_mask = mask
        return value


class _AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()


class _FakeKreaAttention(_FakeKrea):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_AttentionBlock()])


class _AttentionExecutor:
    def __init__(self, model):
        self.class_obj = model

    def __call__(self, x, _timesteps, context, *model_args, **model_kwargs):
        patch = self.class_obj.patch
        rows = (x.shape[-2] + patch - 1) // patch
        cols = (x.shape[-1] + patch - 1) // patch
        sequence = torch.zeros((x.shape[0], context.shape[1] + rows * cols, 1), dtype=x.dtype)
        return self.class_obj.blocks[0].attn(sequence, None, None, transformer_options={})


def test_session_masks_adapter_delta_to_image_tokens_in_region():
    region = Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0)
    loaded = LoadedRegionAdapters("A", "a", 1.0, {"blocks.0.proj": _FakeAdapter()}, [], [])
    model = _FakeKrea()
    session = RouterSession(
        None,
        [region],
        [loaded],
        RouterConfig(feather=0.0),
        token_positions=[[0]],
    )
    output = session.run(
        _Executor(model),
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])},
    )
    assert output.shape == (1, 6, 1)
    assert torch.equal(output[0, :, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0]))


def test_session_preserves_legacy_krea_forward_arguments():
    model = _FakeKrea()
    executor = _Executor(model)
    session = RouterSession(None, [], [], RouterConfig())
    options = {"sigmas": torch.tensor([1.0])}
    attention_mask = torch.ones((1, 2), dtype=torch.bool)
    session.run(
        executor,
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        attention_mask,
        options,
    )
    assert len(executor.model_args) == 2
    assert executor.model_args[0] is attention_mask
    assert executor.model_args[1] is options


def test_session_preserves_new_krea_forward_arguments_with_ref_latents():
    model = _FakeKrea()
    executor = _Executor(model)
    session = RouterSession(None, [], [], RouterConfig())
    options = {"sigmas": torch.tensor([1.0])}
    ref_latents = torch.zeros((1, 16, 2, 2))
    session.run(
        executor,
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        None,
        ref_latents,
        options,
    )
    assert len(executor.model_args) == 3
    assert executor.model_args[0] is None
    assert executor.model_args[1] is ref_latents
    assert executor.model_args[2] is options


def test_session_separates_competing_loras_on_text_and_image_tokens():
    regions = [
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 10.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0),
    ]
    loaded = [
        LoadedRegionAdapters("A", "a", 1.0, {"blocks.0.proj": _FakeAdapter()}, [], []),
        LoadedRegionAdapters("B", "b", 10.0, {"blocks.0.proj": _FakeAdapter()}, [], []),
    ]
    model = _FakeKrea()
    session = RouterSession(
        None,
        regions,
        loaded,
        RouterConfig(feather=0.0),
        token_positions=[[0], [1]],
    )
    output = session.run(
        _Executor(model),
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])},
    )
    assert torch.equal(output[0, :, 0], torch.tensor([1.0, 10.0, 1.0, 10.0, 1.0, 10.0]))


def test_session_routes_canvas_lora_to_unboxed_image_tokens_only():
    character = Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0)
    canvas = Region("Canvas LoRA", True, "style", "", "", 10.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    loaded = [
        LoadedRegionAdapters("A", "a", 1.0, {"blocks.0.proj": _FakeAdapter()}, [], []),
        LoadedRegionAdapters("Canvas LoRA", "style", 10.0, {"blocks.0.proj": _FakeAdapter()}, [], []),
    ]
    session = RouterSession(
        None,
        [character, canvas],
        loaded,
        RouterConfig(feather=0.0),
        token_positions=[[0], []],
        mask_modes=["region", "unboxed"],
        exclusion_regions=[character],
    )
    output = session.run(
        _Executor(_FakeKrea()),
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])},
    )
    assert torch.equal(output[0, :2, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(output[0, 2:, 0], torch.tensor([1.0, 10.0, 1.0, 10.0]))


def test_session_injects_attention_bias_into_krea_attention_mask():
    regions = [
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0),
    ]
    model = _FakeKreaAttention()
    session = RouterSession(
        None,
        regions,
        [],
        RouterConfig(feather=0.0),
        token_positions=[[0], [1]],
    )
    session.run(
        _AttentionExecutor(model),
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={},
    )
    mask = model.blocks[0].attn.received_mask
    assert mask.shape == (1, 1, 6, 6)
    assert float(mask[0, 0, 2, 0]) == 1.0
    assert float(mask[0, 0, 2, 1]) == -5.0


def test_session_matches_krea_padding_for_odd_latent_width():
    regions = [
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0),
    ]
    model = _FakeKreaAttention()
    session = RouterSession(
        None,
        regions,
        [],
        RouterConfig(feather=0.0),
        token_positions=[[0], [1]],
    )
    session.run(
        _AttentionExecutor(model),
        # A five-column latent is padded to six by Krea before 2x2 patching.
        torch.zeros((1, 16, 4, 5)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={},
    )
    mask = model.blocks[0].attn.received_mask
    assert session._grid == (2, 3)
    assert mask.shape == (1, 1, 8, 8)


def test_text_fusion_adapter_is_routed_away_from_competing_subject_tokens():
    regions = [
        Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0),
        Region("B", True, "b", "B", "", 10.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0),
    ]
    session = RouterSession(
        None,
        regions,
        [],
        RouterConfig(feather=0.0),
        token_positions=[[0], [1]],
    )
    session._text_tokens = 3
    session._image_tokens = 4
    session._grid = (2, 2)
    output = torch.zeros((12, 3, 1))
    a_mask = session._token_mask(0, "txtfusion.refiner_blocks.0.attn.wq", 3, 3, output)
    b_mask = session._token_mask(1, "txtfusion.refiner_blocks.0.attn.wq", 3, 3, output)
    assert torch.equal(a_mask[0, :, 0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(b_mask[0, :, 0], torch.tensor([0.0, 1.0, 1.0]))


def test_default_attention_bias_covers_every_krea_block():
    model = _FakeKreaAttention()
    model.blocks = torch.nn.ModuleList([_AttentionBlock() for _ in range(4)])
    session = RouterSession(
        None,
        [],
        [],
        RouterConfig(),
        token_positions=[[0]],
    )
    assert [index for index, _module in session._attention_modules(model)] == [0, 1, 2, 3]


def test_session_uses_external_foreground_mask_instead_of_region_box():
    region = Region("A", True, "a", "A", "", 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0)
    loaded = LoadedRegionAdapters("A", "a", 1.0, {"blocks.0.proj": _FakeAdapter()}, [], [])
    model = _FakeKrea()
    external = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    session = RouterSession(
        None,
        [region],
        [loaded],
        RouterConfig(feather=0.0),
        token_positions=[[0]],
        region_masks=external,
    )
    output = session.run(
        _Executor(model),
        torch.zeros((1, 16, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 2, 8)),
        transformer_options={"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, 0.0])},
    )
    assert torch.equal(output[0, :, 0], torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0]))
