from __future__ import annotations

import torch


def sampling_progress(transformer_options: dict) -> float:
    sigma = transformer_options.get("sigmas")
    if torch.is_tensor(sigma) and sigma.numel():
        current = float(sigma.detach().flatten()[0].float().item())
    elif isinstance(sigma, (int, float)):
        current = float(sigma)
    else:
        return 0.0

    sample_sigmas = transformer_options.get("sample_sigmas")
    if not torch.is_tensor(sample_sigmas) or sample_sigmas.numel() < 2:
        return 0.0
    sigmas = sample_sigmas.detach().float().flatten()
    index = int(torch.argmin((sigmas - current).abs()).item())
    return index / max(1, sigmas.numel() - 1)


def schedule_weight(progress: float, start: float, end: float, softness: float) -> float:
    if progress < start or progress > end or start >= end:
        return 0.0
    softness = max(0.0, min(float(softness), (end - start) / 2.0))
    if softness == 0.0:
        return 1.0

    def smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    fade_in = 1.0 if start <= 0.0 else smoothstep((progress - start) / softness)
    fade_out = 1.0 if end >= 1.0 else smoothstep((end - progress) / softness)
    return min(fade_in, fade_out)
