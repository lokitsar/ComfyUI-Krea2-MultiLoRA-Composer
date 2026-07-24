from __future__ import annotations

import json
import logging

import torch

from .krea2_router.adapters import load_region_adapters
from .krea2_router.config import (
    DEFAULT_CANVAS_LORA_JSON,
    DEFAULT_REGIONS_JSON,
    RouterConfig,
    compose_prompt,
    concept_text_for_canvas,
    concept_text_for_region,
    parse_canvas_lora,
    parse_regions,
)
from .krea2_router.masks import (
    build_region_masks,
    masks_to_preview,
)
from .krea2_router.runtime import RouterSession, install_router_wrapper
from .krea2_router.supersampling import build_supersample_plan, coerce_supersample_plan
from .krea2_router.tokens import find_krea2_concept_positions

LOGGER = logging.getLogger("krea2_multilora_composer")


def _run_common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise,
):
    """Installed-ComfyUI-compatible sampling without importing its top-level nodes module."""
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    batch_indices = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_indices)
    noise_mask = latent.get("noise_mask")
    callback = latent_preview.prepare_callback(model, steps)
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    output = latent.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return output


def _decode_and_downscale(vae, sampled_latent, target_width, target_height, method):
    import comfy.utils

    decoded_latent = sampled_latent["samples"]
    if decoded_latent.is_nested:
        decoded_latent = decoded_latent.unbind()[0]
    working_image = vae.decode(decoded_latent)
    if working_image.ndim == 5:
        working_image = working_image.reshape(
            -1,
            working_image.shape[-3],
            working_image.shape[-2],
            working_image.shape[-1],
        )

    actual_height = int(working_image.shape[1])
    actual_width = int(working_image.shape[2])
    if (actual_width, actual_height) == (target_width, target_height):
        final_image = working_image
    else:
        final_image = comfy.utils.common_upscale(
            working_image.movedim(-1, 1),
            target_width,
            target_height,
            method,
            "disabled",
        ).movedim(1, -1)
    return final_image, working_image


class Krea2CharacterRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "scene_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "Two people in the same scene, natural interaction, coherent lighting",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "regions_json": ("STRING", {"multiline": True, "default": DEFAULT_REGIONS_JSON}),
                "feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "overlap_policy": (["nearest", "normalize", "allow"], {"default": "nearest"}),
                "schedule_softness": (
                    "FLOAT",
                    {"default": 0.04, "min": 0.0, "max": 0.25, "step": 0.01},
                ),
                "strict": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
                # Keep new widgets appended so existing workflow widget values
                # retain their original positional meaning when loaded.
                "supersample_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "canvas_lora_json": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_CANVAS_LORA_JSON},
                ),
            }
        }

    RETURN_TYPES = (
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
    RETURN_NAMES = (
        "model",
        "conditioning",
        "prompt",
        "mask_preview",
        "diagnostics",
        "latent",
        "share_prompt_json",
        "target_width",
        "target_height",
        "supersample_plan",
    )
    FUNCTION = "apply"
    CATEGORY = "Krea2/Multi-LoRA Composer"
    DESCRIPTION = (
        "Routes Krea 2 character LoRAs across image and concept tokens, then applies box-guided "
        "attention bias to reduce identity collisions. Also composes the final spatial prompt."
    )

    def apply(
        self,
        model,
        clip,
        scene_prompt,
        width,
        height,
        regions_json,
        feather,
        overlap_policy,
        schedule_softness,
        strict,
        debug,
        supersample_scale,
        canvas_lora_json=DEFAULT_CANVAS_LORA_JSON,
    ):
        regions, warnings = parse_regions(regions_json)
        canvas_lora, canvas_warnings = parse_canvas_lora(canvas_lora_json)
        warnings.extend(canvas_warnings)
        enabled = [region for region in regions if region.enabled]
        active_characters = [
            region for region in enabled if region.lora not in {"", "None"} and region.strength != 0.0
        ]
        canvas_active = (
            canvas_lora.enabled
            and canvas_lora.lora not in {"", "None"}
            and canvas_lora.strength != 0.0
        )
        routing_regions = list(active_characters)
        mask_modes = ["region"] * len(active_characters)
        if canvas_active:
            routing_regions.append(canvas_lora.as_region())
            mask_modes.append(canvas_lora.coverage)
        if strict and not routing_regions:
            raise ValueError("No enabled character or Canvas LoRA has a selected LoRA")

        prompt = compose_prompt(scene_prompt, enabled, canvas_lora if canvas_active else None)
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        concept_texts = [concept_text_for_region(region) for region in active_characters]
        concept_required = [True] * len(active_characters)
        fallbacks = [region.trigger for region in active_characters]
        if canvas_active:
            canvas_concept = concept_text_for_canvas(canvas_lora)
            concept_texts.append(canvas_concept)
            concept_required.append(bool(canvas_concept))
            fallbacks.append(canvas_lora.trigger)
        token_positions: list[list[int]] = [[] for _ in routing_regions]
        used_concepts = concept_texts
        try:
            token_positions, used_concepts = find_krea2_concept_positions(
                clip,
                tokens,
                concept_texts,
                fallbacks,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Could not align routed LoRA phrases to Krea2 tokens: {exc}") from exc
            warnings.append(f"Token routing disabled: {exc}")
        missing_tokens = [
            routing_regions[index].name
            for index, positions in enumerate(token_positions)
            if concept_required[index] and not positions
        ]
        if missing_tokens:
            message = "No Krea2 token positions found for: " + ", ".join(missing_tokens)
            if strict:
                raise ValueError(message)
            warnings.append(message)

        config = RouterConfig(
            feather=float(feather),
            overlap_policy=str(overlap_policy),
            schedule_softness=float(schedule_softness),
            strict=bool(strict),
            debug=bool(debug),
        )
        loaded = []
        for region in routing_regions:
            result = load_region_adapters(model, region, strict=bool(strict))
            loaded.append(result)
            if result.skipped_keys:
                warnings.append(
                    f"{region.name}: skipped {len(result.skipped_keys)} non-spatial LoRA targets"
                )
            if result.unsupported:
                warnings.append(
                    f"{region.name}: skipped {len(result.unsupported)} unsupported adapter targets"
                )

        patched = model.clone()
        if loaded:
            session = RouterSession(
                patched,
                routing_regions,
                loaded,
                config,
                token_positions=token_positions,
                mask_modes=mask_modes,
                exclusion_regions=enabled,
            )
            install_router_wrapper(patched, session)

        preview_rows = max(16, int(height) // 8)
        preview_cols = max(16, int(width) // 8)
        preview_masks = build_region_masks(
            preview_rows,
            preview_cols,
            enabled,
            config.feather,
            config.overlap_policy,
        )
        preview = masks_to_preview(preview_masks)
        supersample_plan = build_supersample_plan(width, height, supersample_scale)
        supersampling = supersample_plan.as_dict()
        diagnostics = {
            "version": "0.5.0",
            "engine": "box_guided_token_routing_attention_bias",
            "prompt": prompt,
            "canvas": {"width": int(width), "height": int(height)},
            "supersampling": supersampling,
            "regions": [
                {
                    "name": region.name,
                    "lora": region.lora,
                    "trigger": region.trigger,
                    "strength": region.strength,
                    "box": list(region.box),
                    "schedule": [region.start, region.end],
                    "mask_mode": mask_modes[index],
                    "concept_text": used_concepts[index],
                    "token_positions": token_positions[index],
                    "matched_modules": len(result.adapters),
                    "skipped_modules": len(result.skipped_keys),
                }
                for index, (region, result) in enumerate(zip(routing_regions, loaded, strict=True))
            ],
            "attention_bias": {
                "enabled": bool(config.attention_bias and any(token_positions)),
                "negative": config.negative_bias,
                "positive": config.positive_bias,
                "blocks": "all",
            },
            "warnings": warnings,
        }
        share_prompt = {
            "format": "krea2_character_router_share_v1",
            "canvas": {"width": int(width), "height": int(height)},
            "supersampling": supersampling,
            "scene_prompt": str(scene_prompt).strip(),
            "composed_positive_prompt": prompt,
            "router": {
                "feather": config.feather,
                "overlap_policy": config.overlap_policy,
                "schedule_softness": config.schedule_softness,
                "strict": config.strict,
            },
            "canvas_lora": {
                "enabled": canvas_lora.enabled,
                "lora": canvas_lora.lora,
                "trigger": canvas_lora.trigger,
                "description": canvas_lora.prompt,
                "strength": canvas_lora.strength,
                "coverage": canvas_lora.coverage,
                "schedule": {"start": canvas_lora.start, "end": canvas_lora.end},
            },
            "characters": [
                {
                    "name": region.name,
                    "enabled": region.enabled,
                    "lora": region.lora,
                    "trigger": region.trigger,
                    "description": region.prompt,
                    "strength": region.strength,
                    "placement": {
                        "normalized": {
                            "x": region.x,
                            "y": region.y,
                            "width": region.w,
                            "height": region.h,
                        },
                        "pixels": {
                            "x": round(region.x * int(width)),
                            "y": round(region.y * int(height)),
                            "width": round(region.w * int(width)),
                            "height": round(region.h * int(height)),
                        },
                    },
                    "schedule": {"start": region.start, "end": region.end},
                }
                for region in regions
            ],
            "warnings": warnings,
        }
        import comfy.model_management

        latent = torch.zeros(
            [
                1,
                4,
                supersample_plan.working_height // 8,
                supersample_plan.working_width // 8,
            ],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        latent_output = {"samples": latent, "downscale_ratio_spacial": 8}
        if debug:
            LOGGER.info("Krea2 Multi-LoRA Composer diagnostics: %s", diagnostics)
        return (
            patched,
            conditioning,
            prompt,
            preview,
            json.dumps(diagnostics, indent=2),
            latent_output,
            json.dumps(share_prompt, indent=2),
            supersample_plan.target_width,
            supersample_plan.target_height,
            supersample_plan,
        )


class Krea2SupersampledKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "supersample_plan": ("KREA2_SUPERSAMPLE_PLAN",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "downscale_method": (
                    ["lanczos", "bicubic", "area", "bilinear"],
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "latent", "working_image", "diagnostics")
    FUNCTION = "sample"
    CATEGORY = "Krea2/Multi-LoRA Composer"
    DESCRIPTION = (
        "Samples the router's larger working latent, decodes it, and automatically downsizes the "
        "result to the router's original target resolution."
    )

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        vae,
        supersample_plan,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        downscale_method,
    ):
        plan = coerce_supersample_plan(supersample_plan)
        latent_samples = latent_image.get("samples")
        if not isinstance(latent_samples, torch.Tensor) or latent_samples.ndim < 4:
            raise ValueError("latent_image does not contain a valid samples tensor")

        spatial_ratio = int(latent_image.get("downscale_ratio_spacial", 8))
        expected_rows = plan.working_height // spatial_ratio
        expected_cols = plan.working_width // spatial_ratio
        actual_rows, actual_cols = int(latent_samples.shape[-2]), int(latent_samples.shape[-1])
        if (actual_rows, actual_cols) != (expected_rows, expected_cols):
            raise ValueError(
                "The latent resolution does not match the supersample plan: "
                f"expected {plan.working_width}x{plan.working_height}, "
                f"received {actual_cols * spatial_ratio}x{actual_rows * spatial_ratio}. "
                "Connect latent and supersample_plan from the same Krea2 Multi-LoRA Composer."
            )

        sampled_latent = _run_common_ksampler(
            model,
            int(seed),
            int(steps),
            float(cfg),
            str(sampler_name),
            str(scheduler),
            positive,
            negative,
            latent_image,
            float(denoise),
        )

        final_image, working_image = _decode_and_downscale(
            vae,
            sampled_latent,
            plan.target_width,
            plan.target_height,
            str(downscale_method),
        )
        actual_height = int(working_image.shape[1])
        actual_width = int(working_image.shape[2])

        diagnostics = {
            "version": "0.5.0",
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler": str(sampler_name),
            "scheduler": str(scheduler),
            "denoise": float(denoise),
            "downscale_method": str(downscale_method),
            "target": [plan.target_width, plan.target_height],
            "working": [actual_width, actual_height],
            "requested_scale": plan.scale,
        }
        return (
            final_image.clamp(0.0, 1.0),
            sampled_latent,
            working_image.clamp(0.0, 1.0),
            json.dumps(diagnostics, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "Krea2CharacterRouter": Krea2CharacterRouter,
    "Krea2SupersampledKSampler": Krea2SupersampledKSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2CharacterRouter": "Krea2 Multi-LoRA Composer",
    "Krea2SupersampledKSampler": "Krea2 Multi-LoRA SuperSampler",
}
