from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from typing import Any

import torch


ROUTEABLE_PREFIXES = ("blocks.", "first", "last.linear", "txtfusion.")


@dataclass
class LoadedRegionAdapters:
    name: str
    lora: str
    strength: float
    adapters: dict[str, Any]
    skipped_keys: list[str]
    unsupported: list[str]


def module_path_from_target(target_key: str) -> str | None:
    prefix = "diffusion_model."
    if not target_key.startswith(prefix) or not target_key.endswith(".weight"):
        return None
    path = target_key[len(prefix) : -len(".weight")]
    return path if path.startswith(ROUTEABLE_PREFIXES) else None


def _move_value(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        target_dtype = dtype if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _move_value(item, device, dtype) for key, item in value.items()}
    return value


def prepare_adapter(adapter: Any, module: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> Any:
    """Clone a native ComfyUI adapter and move only its small adapter weights."""
    prepared = copy.copy(adapter)
    if hasattr(adapter, "weights"):
        prepared.weights = _move_value(adapter.weights, device, dtype)
    prepared.shape = tuple(getattr(module, "weight").shape)
    prepared.is_conv = False
    prepared.conv_dim = 0
    prepared.kw_dict = {}
    return prepared


def load_region_adapters(model, region, strict: bool) -> LoadedRegionAdapters:
    """Load Krea adapters that can be routed by image or subject-token position."""
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path("loras", region.lora)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"LoRA not found: {region.lora}")

    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    converted = comfy.lora_convert.convert_lora(state_dict)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)

    adapters: dict[str, Any] = {}
    skipped: list[str] = []
    unsupported: list[str] = []
    for target_key, patch in patches.items():
        module_path = module_path_from_target(target_key)
        if module_path is None:
            skipped.append(target_key)
            continue
        if not hasattr(patch, "h") or not hasattr(patch, "g"):
            unsupported.append(target_key)
            continue
        adapters[module_path] = patch

    if strict and not adapters:
        detail = " no Krea transformer adapters were matched"
        if unsupported:
            detail += f"; {len(unsupported)} matched patches use unsupported formats"
        raise ValueError(f"{region.name}:{detail}")
    return LoadedRegionAdapters(region.name, region.lora, region.strength, adapters, skipped, unsupported)
