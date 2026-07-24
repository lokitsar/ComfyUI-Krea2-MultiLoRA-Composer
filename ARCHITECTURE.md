# Architecture

## Scope

Krea2 Multi-LoRA Composer performs single-pass, token-aware spatial routing of independently trained Krea 2
adapters. It does not merge adapter weights, modify the Krea 2 checkpoint, perform automatic subject discovery,
or provide reference-image conditioning.

## Runtime pipeline

1. Parse and validate normalized character regions and the optional Canvas LoRA.
2. Resolve each LoRA through ComfyUI's Krea-specific model key map.
3. Compose the scene, optional Canvas phrase, and character descriptions into one positive prompt.
4. Match each routed phrase to Krea 2's post-template conditioning-token positions.
5. Clone the model and install one keyed diffusion-model wrapper.
6. Map adapter target paths to live modules on the first model call.
7. Route adapter contributions by image region and subject-token ownership.
8. Apply subject-to-region attention bias across Krea transformer blocks.
9. Remove temporary PyTorch hooks after every model call.
10. Release cached device tensors through ComfyUI's cleanup callback.

## Routeable adapter targets

- `blocks.*` shared transformer projections
- `first` image-token projection
- `last.linear` output projection
- `txtfusion.*` text-only projections

Text-fusion targets are routed by subject-token position. Timestep and other global targets are skipped because a
2D region or subject-token mask has no defined meaning for them. Strict mode rejects a LoRA with no routeable
Krea targets or a character phrase that cannot be aligned to the conditioning sequence.

## Image-token routing

Regions are stored in normalized image coordinates. At runtime, masks are resized to the actual Krea token grid
derived from the latent and model patch size. Feathering occurs inward from each region edge, so direct adapter
contributions remain exactly zero outside the assigned region.

The optional Canvas LoRA supports two mask modes:

- `unboxed`: the complement of the union of all enabled character masks
- `global`: a mask of ones covering the entire token grid

The complement is calculated after character feathering, producing a shared transition where character and
Canvas contributions exchange strength rather than forming a hard rectangular seam.

When regions overlap:

- `nearest` assigns each overlapping token to its closest region center.
- `normalize` divides the overlapping contribution between regions.
- `allow` retains both contributions.

## Subject-token isolation

Each routed adapter's text-fusion contribution is retained on shared text and its own phrase while being
suppressed on competing routed phrases. In `unboxed` mode, the Canvas LoRA is more restrictive: its text-fusion
contribution is kept only on its own Canvas phrase and removed from shared scene tokens. A triggerless,
description-free Canvas LoRA therefore has no text-fusion contribution, while its spatial adapter targets still
operate outside the character boxes. `global` mode retains shared-text behavior. This prevents global LoRA
stacking from giving multiple identities the same subject-token ownership.

## Attention bias

The model wrapper adds attention bias that favors a subject phrase's own image region and suppresses that subject
outside its region, including unassigned gaps. This is a model guidance mechanism, not an absolute segmentation
boundary; Krea's base attention remains scene-wide.

## Transparent supersampling

The Composer produces a `SupersamplePlan` containing:

- Original target width and height
- Internal working width and height
- Requested scale
- Contract version

The Multi-LoRA SuperSampler validates that its latent matches the plan, samples at the working resolution,
VAE-decodes the result, and downsizes the decoded image to the original target. Downscaling occurs in image space
so the final output benefits from high-resolution denoising without requiring a second diffusion pass.

## Backward compatibility

The public project was renamed in version `0.4.0`. Internal ComfyUI identifiers and the
`krea2_character_router_share_v1` scene format remain unchanged so existing workflows and shared scenes continue
to load. Version `0.5.0` appends the optional Canvas LoRA widget and adds an optional `canvas_lora` object to
scene JSON without changing the format identifier.

## Planned extensions

- Optional face-focus routing inside a broader character region
- Post-generation face or full-character refinement
- Pose-control composition after stable Krea 2 pose Control-LoRA support is validated
