from __future__ import annotations

import json
import logging

import torch

from .krea2_router.adapters import load_region_adapters
from .krea2_router.config import (
    DEFAULT_REGIONS_JSON,
    RouterConfig,
    compose_prompt,
    concept_text_for_region,
    parse_regions,
)
from .krea2_router.masks import (
    build_region_masks,
    masks_to_preview,
    prepare_external_masks,
    resize_mask_batch,
    resolve_mask_overlaps,
)
from .krea2_router.plan import RoutePlan, sam3_box_prompt
from .krea2_router.runtime import RouterSession, install_router_wrapper
from .krea2_router.sam_masks import assign_detections_to_regions
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
    ):
        regions, warnings = parse_regions(regions_json)
        enabled = [region for region in regions if region.enabled]
        active = [region for region in enabled if region.lora not in {"", "None"} and region.strength != 0.0]
        if strict and not active:
            raise ValueError("No enabled character region has a selected LoRA")

        prompt = compose_prompt(scene_prompt, enabled)
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        concept_texts = [concept_text_for_region(region) for region in active]
        token_positions: list[list[int]] = [[] for _ in active]
        used_concepts = concept_texts
        try:
            token_positions, used_concepts = find_krea2_concept_positions(
                clip,
                tokens,
                concept_texts,
                [region.trigger for region in active],
            )
        except (AttributeError, TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Could not align character phrases to Krea2 tokens: {exc}") from exc
            warnings.append(f"Token routing disabled: {exc}")
        missing_tokens = [active[index].name for index, positions in enumerate(token_positions) if not positions]
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
        for region in active:
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
            session = RouterSession(patched, active, loaded, config, token_positions=token_positions)
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
            "version": "0.3.1",
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
                    "concept_text": used_concepts[index],
                    "token_positions": token_positions[index],
                    "matched_modules": len(result.adapters),
                    "skipped_modules": len(result.skipped_keys),
                }
                for index, (region, result) in enumerate(zip(active, loaded))
            ],
            "attention_bias": {
                "enabled": bool(config.attention_bias and all(token_positions)),
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
            "version": "0.4.0",
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


class Krea2SAM3RegionBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "route_plan": ("KREA2_ROUTE_PLAN",),
                "character_index": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "padding": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAM3_BOXES_PROMPT", "SAM3_PROMPT_PIPELINE", "STRING")
    RETURN_NAMES = ("sam3_box", "tbg_pipeline", "region")
    FUNCTION = "select"
    CATEGORY = "Krea2/Multi-LoRA Composer"
    DESCRIPTION = "Converts one existing Krea2 character region into a box prompt for SAM3."

    def select(self, route_plan, character_index, padding):
        prompt, region = sam3_box_prompt(route_plan, int(character_index), float(padding))
        tbg_pipeline = {
            "positive_points": None,
            "negative_points": None,
            "positive_boxes": prompt,
            "negative_boxes": None,
        }
        detail = {
            "character_index": int(character_index),
            "name": region.name,
            "box_center": prompt["boxes"][0],
        }
        return (prompt, tbg_pipeline, json.dumps(detail, indent=2))


class Krea2SAMMaskRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "route_plan": ("KREA2_ROUTE_PLAN",),
                "identity_masks": ("MASK",),
                "mask_grow": ("INT", {"default": 8, "min": -64, "max": 128, "step": 1}),
                "mask_feather": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "conditioning", "prompt", "mask_preview", "diagnostics")
    FUNCTION = "apply"
    CATEGORY = "Krea2/Multi-LoRA Composer"
    DESCRIPTION = (
        "Second-pass router that confines each identity LoRA and its subject tokens to an external "
        "SAM3 foreground mask. MASK batch order must match Character A, B, C."
    )

    def apply(self, model, route_plan, identity_masks, mask_grow, mask_feather, debug):
        if not isinstance(route_plan, RoutePlan):
            raise TypeError("route_plan must come from Krea2 Multi-LoRA Composer")
        count = len(route_plan.regions)
        if count == 0:
            raise ValueError("route plan has no active LoRA characters")
        prepared_masks = prepare_external_masks(
            identity_masks,
            expected=count,
            grow=int(mask_grow),
            feather=int(mask_feather),
        ).cpu()

        config = RouterConfig(
            feather=0.0,
            overlap_policy=route_plan.config.overlap_policy,
            schedule_softness=route_plan.config.schedule_softness,
            attention_bias=route_plan.config.attention_bias,
            negative_bias=route_plan.config.negative_bias,
            positive_bias=route_plan.config.positive_bias,
            bias_block_fraction=route_plan.config.bias_block_fraction,
            strict=route_plan.config.strict,
            debug=bool(debug),
        )
        patched = model.clone()
        session = RouterSession(
            patched,
            list(route_plan.regions),
            list(route_plan.loaded),
            config,
            token_positions=[list(positions) for positions in route_plan.token_positions],
            region_masks=prepared_masks,
        )
        install_router_wrapper(patched, session)

        preview_rows = max(16, route_plan.height // 8)
        preview_cols = max(16, route_plan.width // 8)
        preview_masks = resize_mask_batch(prepared_masks, preview_rows, preview_cols)
        preview_masks = resolve_mask_overlaps(
            preview_masks,
            route_plan.regions,
            config.overlap_policy,
        )
        preview = masks_to_preview(preview_masks)
        diagnostics = {
            "version": "0.3.3",
            "engine": "sam3_foreground_mask_refinement",
            "characters": [region.name for region in route_plan.regions],
            "mask_batch": list(prepared_masks.shape),
            "mask_grow": int(mask_grow),
            "mask_feather": int(mask_feather),
            "overlap_policy": config.overlap_policy,
            "attention_bias": "external masks",
            "warnings": list(route_plan.warnings),
        }
        if debug:
            LOGGER.info("Krea2 SAM Mask Refiner diagnostics: %s", diagnostics)
        return (
            patched,
            route_plan.conditioning,
            route_plan.prompt,
            preview,
            json.dumps(diagnostics, indent=2),
        )


class Krea2SAM3AutoMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "route_plan": ("KREA2_ROUTE_PLAN",),
                "confidence_threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("identity_masks", "mask_preview", "diagnostics")
    FUNCTION = "segment"
    CATEGORY = "Krea2/Multi-LoRA Composer"
    DESCRIPTION = (
        "Runs one TBG SAM3 image pass for all saved character boxes, assigns detections by box IoU, "
        "and returns masks already ordered as Character A, B, C."
    )

    def segment(self, sam3_model, image, route_plan, confidence_threshold):
        if not isinstance(route_plan, RoutePlan):
            raise TypeError("route_plan must come from Krea2 Multi-LoRA Composer")
        if not isinstance(sam3_model, dict) or "model" not in sam3_model or "processor" not in sam3_model:
            raise TypeError("sam3_model must come from TBG SAM3 Model Loader")
        if not route_plan.regions:
            raise ValueError("route plan has no active characters")

        import numpy as np
        from PIL import Image

        source = image[0].detach().float().cpu().clamp(0.0, 1.0)
        height, width = int(source.shape[0]), int(source.shape[1])
        pil_image = Image.fromarray((source.numpy() * 255.0).round().astype(np.uint8))

        model = sam3_model["model"]
        processor = sam3_model["processor"]
        target_device = str(sam3_model.get("original_device", sam3_model.get("device", "cuda")))
        current_device = str(next(model.parameters()).device)
        if current_device != target_device:
            model.to(target_device)
            processor.device = target_device
            sam3_model["device"] = target_device

        processor.set_confidence_threshold(float(confidence_threshold))
        state = processor.set_image(pil_image)
        boxes = [[*region.center, region.w, region.h] for region in route_plan.regions]
        state = processor.add_multiple_box_prompts(boxes, [True] * len(boxes), state)
        raw_masks = state.get("masks")
        raw_boxes = state.get("boxes")
        raw_scores = state.get("scores")
        if raw_masks is None or len(raw_masks) == 0:
            raise ValueError(
                "SAM3 found no character masks. Lower confidence_threshold or enlarge the saved boxes slightly."
            )

        masks = torch.as_tensor(raw_masks).float()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(f"SAM3 returned unsupported mask shape {tuple(masks.shape)}")
        if raw_boxes is None:
            if masks.shape[0] != len(route_plan.regions):
                raise ValueError(
                    f"SAM3 returned {masks.shape[0]} masks without boxes for {len(route_plan.regions)} characters"
                )
            selected = list(range(len(route_plan.regions)))
            overlaps = [0.0] * len(selected)
        else:
            selected, overlaps = assign_detections_to_regions(
                raw_boxes,
                route_plan.regions,
                width,
                height,
                raw_scores,
            )
        ordered = masks[selected].detach().cpu().clamp(0.0, 1.0)
        if ordered.shape[-2:] != (height, width):
            ordered = torch.nn.functional.interpolate(
                ordered.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(1)

        colors = torch.tensor(
            [[0.95, 0.25, 0.30], [0.20, 0.55, 1.00], [0.20, 0.85, 0.45], [0.95, 0.70, 0.20]],
            dtype=source.dtype,
        )
        preview = source.clone()
        for index, mask in enumerate(ordered):
            alpha = (mask.clamp(0.0, 1.0) * 0.45).unsqueeze(-1)
            color = colors[index % len(colors)]
            preview = preview * (1.0 - alpha) + color * alpha
        diagnostics = {
            "version": "0.3.3",
            "characters": [region.name for region in route_plan.regions],
            "detections": int(masks.shape[0]),
            "selected_indices": selected,
            "box_iou": overlaps,
            "mask_shape": list(ordered.shape),
        }
        return (ordered, preview.unsqueeze(0).clamp(0.0, 1.0), json.dumps(diagnostics, indent=2))


NODE_CLASS_MAPPINGS = {
    "Krea2CharacterRouter": Krea2CharacterRouter,
    "Krea2SupersampledKSampler": Krea2SupersampledKSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2CharacterRouter": "Krea2 Multi-LoRA Composer",
    "Krea2SupersampledKSampler": "Krea2 Multi-LoRA SuperSampler",
}
