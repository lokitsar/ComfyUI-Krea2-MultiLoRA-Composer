from __future__ import annotations

import logging
from typing import Any

import torch

from .adapters import LoadedRegionAdapters, prepare_adapter
from .attention import construct_box_attention_bias
from .config import Region, RouterConfig
from .masks import build_routing_masks, resize_mask_batch, resolve_mask_overlaps
from .schedule import sampling_progress, schedule_weight

LOGGER = logging.getLogger("krea2_multilora_composer")
WRAPPER_KEY = "krea2_character_router_v0"


def is_krea2_diffusion_model(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("blocks", "txtfusion", "txtmlp", "_unpack_context", "patch"))


def _resolve_diffusion_model(patcher) -> Any:
    base_model = patcher.model
    return getattr(base_model, "diffusion_model", base_model)


class RouterSession:
    def __init__(
        self,
        patcher,
        regions: list[Region],
        loaded: list[LoadedRegionAdapters],
        config: RouterConfig,
        token_positions: list[list[int]] | None = None,
        region_masks: torch.Tensor | None = None,
        mask_modes: list[str] | None = None,
        exclusion_regions: list[Region] | None = None,
    ):
        self.patcher = patcher
        self.regions = regions
        self.loaded = loaded
        self.config = config
        self.token_positions = token_positions or []
        self.region_masks = region_masks
        self.mask_modes = mask_modes or ["region"] * len(regions)
        self.exclusion_regions = exclusion_regions
        self._module_map: dict[str, torch.nn.Module] | None = None
        self._prepared: dict[str, list[tuple[int, Any]]] | None = None
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._attention_bias_cache: dict[tuple, torch.Tensor | None] = {}
        self._text_tokens = 0
        self._image_tokens = 0
        self._progress = 0.0
        self._grid = (1, 1)
        self._debug_reported = False
        self._debug_energy: dict[int, dict[str, Any]] = {}

    def _find_modules(self, diffusion_model) -> dict[str, torch.nn.Module]:
        available = dict(diffusion_model.named_modules())
        requested = {path for region in self.loaded for path in region.adapters}
        missing = sorted(requested - available.keys())
        if missing and self.config.strict:
            raise ValueError(f"Krea 2 modules not found for {len(missing)} LoRA targets: {missing[:4]}")
        return {path: available[path] for path in requested if path in available}

    def cleanup(self) -> None:
        """Release device-resident adapter copies when ComfyUI cleans up the model."""
        self._prepared = None
        self._module_map = None
        self._mask_cache.clear()
        self._attention_bias_cache.clear()

    def _prepare(self, diffusion_model, device: torch.device, dtype: torch.dtype) -> None:
        self._module_map = self._find_modules(diffusion_model)
        prepared: dict[str, list[tuple[int, Any]]] = {}
        for region_index, loaded_region in enumerate(self.loaded):
            for path, adapter in loaded_region.adapters.items():
                module = self._module_map.get(path)
                if module is None:
                    continue
                prepared_adapter = prepare_adapter(adapter, module, device, dtype)
                prepared.setdefault(path, []).append((region_index, prepared_adapter))
        self._prepared = prepared

    def _masks(self, rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (
            rows,
            cols,
            str(device),
            dtype,
            self.config.feather,
            self.config.overlap_policy,
            tuple(self.mask_modes),
        )
        cached = self._mask_cache.get(key)
        if cached is None:
            if self.region_masks is None:
                cached = build_routing_masks(
                    rows,
                    cols,
                    self.regions,
                    self.config.feather,
                    self.config.overlap_policy,
                    self.mask_modes,
                    self.exclusion_regions,
                ).to(device=device, dtype=dtype)
            else:
                cached = resize_mask_batch(self.region_masks, rows, cols, device, dtype)
                cached = resolve_mask_overlaps(cached, self.regions, self.config.overlap_policy)
            self._mask_cache[key] = cached
        return cached

    def _token_mask(self, region_index: int, path: str, sequence: int, ndim: int, output: torch.Tensor) -> torch.Tensor:
        rows = max(1, round(self._image_tokens**0.5))
        cols = max(1, self._image_tokens // rows)
        # Non-square canvases cannot be recovered from token count alone. The wrapper
        # stores the exact grid before hooks are installed.
        rows, cols = self._grid
        mask = self._masks(rows, cols, output.device, output.dtype)[region_index].reshape(-1)
        base = torch.zeros(sequence, device=output.device, dtype=output.dtype)
        if path == "first" and sequence == self._image_tokens:
            base[:] = mask
        else:
            # FreeFuse-style token routing: an adapter may influence shared text and
            # its own subject phrase, but never a competing character's phrase.
            if self.token_positions:
                text_stop = min(sequence, self._text_tokens)
                if self.mask_modes[region_index] == "unboxed":
                    # Keep an unboxed style LoRA off shared scene tokens so its
                    # text-fusion path cannot restyle boxed characters indirectly.
                    for position in self.token_positions[region_index]:
                        if 0 <= position < text_stop:
                            base[position] = 1
                else:
                    base[:text_stop] = 1
                    for other_index, positions in enumerate(self.token_positions):
                        if other_index == region_index:
                            continue
                        for position in positions:
                            if 0 <= position < text_stop:
                                base[position] = 0
            start = self._text_tokens
            stop = min(sequence, start + self._image_tokens)
            base[start:stop] = mask[: max(0, stop - start)]
        return base.view(*([1] * (ndim - 2)), sequence, 1)

    def _attention_bias(self, batch: int, sequence: int, output: torch.Tensor) -> torch.Tensor | None:
        if not self.config.attention_bias or not self.token_positions:
            return None
        key = (
            batch,
            sequence,
            self._grid,
            str(output.device),
            output.dtype,
            self.config.negative_bias,
            self.config.positive_bias,
        )
        if key not in self._attention_bias_cache:
            rows, cols = self._grid
            masks = self._masks(rows, cols, output.device, output.dtype)
            bias = construct_box_attention_bias(
                masks,
                self.token_positions,
                self._text_tokens,
                negative_bias=self.config.negative_bias,
                positive_bias=self.config.positive_bias,
                bidirectional=True,
            )
            if bias is not None and bias.shape[-1] != sequence:
                message = (
                    f"Krea2 attention sequence mismatch: built {bias.shape[-1]} tokens, "
                    f"attention received {sequence}"
                )
                if self.config.strict:
                    raise ValueError(message)
                LOGGER.warning(message)
                bias = None
            if bias is not None and batch > 1:
                bias = bias.expand(batch, -1, -1)
            self._attention_bias_cache[key] = bias
        return self._attention_bias_cache[key]

    def _make_attention_bias_hook(self, block_index: int):
        def prehook(_module, args, kwargs):
            if not args or not torch.is_tensor(args[0]):
                return None
            source = args[0]
            bias = self._attention_bias(source.shape[0], source.shape[1], source)
            if bias is None:
                return None
            bias = bias.unsqueeze(1)
            new_args = list(args)
            new_kwargs = dict(kwargs or {})
            existing = new_args[2] if len(new_args) >= 3 else new_kwargs.get("mask")
            if torch.is_tensor(existing):
                if existing.dtype == torch.bool:
                    message = f"Cannot combine boolean Krea2 attention mask at block {block_index}"
                    if self.config.strict:
                        raise ValueError(message)
                    LOGGER.warning(message)
                    return None
                bias = bias + existing
            if len(new_args) >= 3:
                new_args[2] = bias
            else:
                new_kwargs["mask"] = bias
            return tuple(new_args), new_kwargs

        return prehook

    def _attention_modules(self, diffusion_model) -> list[tuple[int, torch.nn.Module]]:
        if not self.config.attention_bias or not self.token_positions:
            return []
        blocks = getattr(diffusion_model, "blocks", [])
        start = int(len(blocks) * self.config.bias_block_fraction)
        return [
            (index, block.attn)
            for index, block in enumerate(blocks)
            if index >= start and hasattr(block, "attn")
        ]

    def _record_debug_energy(
        self,
        region_index: int,
        raw_delta: torch.Tensor,
        routed_delta: torch.Tensor,
    ) -> None:
        if not self.config.debug or self._debug_reported:
            return
        stats = self._debug_energy.setdefault(
            region_index,
            {"raw": raw_delta.new_zeros((), dtype=torch.float32),
             "routed": raw_delta.new_zeros((), dtype=torch.float32), "calls": 0},
        )
        stats["raw"] = stats["raw"] + raw_delta.detach().float().square().mean()
        stats["routed"] = stats["routed"] + routed_delta.detach().float().square().mean()
        stats["calls"] += 1

    def _report_debug_energy(self, biased_blocks: list[tuple[int, torch.nn.Module]]) -> None:
        if not self.config.debug or self._debug_reported:
            return
        report = []
        for region_index, stats in sorted(self._debug_energy.items()):
            calls = max(1, int(stats["calls"]))
            report.append(
                {
                    "region": self.regions[region_index].name,
                    "raw_delta_rms": float((stats["raw"] / calls).sqrt().item()),
                    "routed_delta_rms": float((stats["routed"] / calls).sqrt().item()),
                    "module_calls": calls,
                    "concept_tokens": self.token_positions[region_index]
                    if region_index < len(self.token_positions)
                    else [],
                }
            )
        LOGGER.info(
            "Krea2 Multi-LoRA Composer runtime proof: grid=%s biased_blocks=%s adapters=%s",
            self._grid,
            [index for index, _module in biased_blocks],
            report,
        )
        self._debug_reported = True
        self._debug_energy.clear()

    def _make_hook(self, path: str, entries: list[tuple[int, Any]]):
        def hook(_module, inputs, output):
            if not torch.is_tensor(output) or output.ndim < 2 or not inputs or not torch.is_tensor(inputs[0]):
                return output
            source = inputs[0]
            routed_delta = None
            for region_index, adapter in entries:
                region = self.regions[region_index]
                weight = schedule_weight(
                    self._progress, region.start, region.end, self.config.schedule_softness
                )
                if weight == 0.0 or region.strength == 0.0:
                    continue
                adapter.multiplier = float(region.strength) * weight
                delta = adapter.g(output + adapter.h(source, output)) - output
                routed = delta * self._token_mask(
                    region_index, path, output.shape[-2], output.ndim, output
                )
                self._record_debug_energy(region_index, delta, routed)
                routed_delta = routed if routed_delta is None else routed_delta + routed
            return output if routed_delta is None else output + routed_delta.to(output.dtype)

        return hook

    def run(
        self,
        executor,
        x,
        timesteps,
        context,
        *model_args,
        **model_kwargs,
    ):
        transformer_options = model_kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            # ComfyUI has shipped Krea 2 forwards both with and without a
            # ref_latents positional argument. In both forms transformer_options
            # is the final explicit dict. Read it without rebuilding the call.
            transformer_options = next(
                (value for value in reversed(model_args) if isinstance(value, dict)),
                {},
            )
        diffusion_model = executor.class_obj
        if not is_krea2_diffusion_model(diffusion_model):
            message = "Krea2 Multi-LoRA Composer received a non-Krea2 diffusion model"
            if self.config.strict:
                raise ValueError(message)
            LOGGER.warning(message)
            return executor(x, timesteps, context, *model_args, **model_kwargs)

        self._text_tokens = int(context.shape[1])
        patch = int(getattr(diffusion_model, "patch", 2))
        latent_rows, latent_cols = int(x.shape[-2]), int(x.shape[-1])
        # Krea2 pads odd latent dimensions up to a full patch before flattening
        # image tokens. Use the same ceiling division or the final token column/
        # row will be missing from both LoRA masks and the attention-bias matrix.
        self._grid = (
            max(1, (latent_rows + patch - 1) // patch),
            max(1, (latent_cols + patch - 1) // patch),
        )
        self._image_tokens = self._grid[0] * self._grid[1]
        self._progress = sampling_progress(transformer_options)
        if self._prepared is None:
            self._prepare(diffusion_model, x.device, x.dtype)

        handles = []
        biased_blocks = self._attention_modules(diffusion_model)
        try:
            for path, entries in self._prepared.items():
                handles.append(self._module_map[path].register_forward_hook(self._make_hook(path, entries)))
            for block_index, attention in biased_blocks:
                handles.append(
                    attention.register_forward_pre_hook(
                        self._make_attention_bias_hook(block_index), with_kwargs=True
                    )
                )
            result = executor(x, timesteps, context, *model_args, **model_kwargs)
            self._report_debug_energy(biased_blocks)
            return result
        finally:
            for handle in handles:
                handle.remove()


def install_router_wrapper(patched_model, session: RouterSession) -> None:
    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLEANUP
    if hasattr(patched_model, "remove_wrappers_with_key"):
        patched_model.remove_wrappers_with_key(wrapper_type, WRAPPER_KEY)
    if hasattr(patched_model, "remove_callbacks_with_key"):
        patched_model.remove_callbacks_with_key(callback_type, WRAPPER_KEY)

    def wrapper(executor, x, timesteps, context, *model_args, **model_kwargs):
        return session.run(executor, x, timesteps, context, *model_args, **model_kwargs)

    patched_model.add_wrapper_with_key(wrapper_type, WRAPPER_KEY, wrapper)
    patched_model.add_callback_with_key(callback_type, WRAPPER_KEY, lambda _patcher: session.cleanup())
