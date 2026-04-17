"""Optional synthetic degradations used for evaluation and debugging.

These help you verify that the policy "knows" how to brighten / de-gamma an
image, independent of real-world distributions. Training uses unpaired
natural data directly; these are *not* required for the main loop.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def lower_gamma(x: Tensor, gamma: float = 0.4) -> Tensor:
    """Darken via gamma < 1 (pushes toward blacks). Returns [B, C, H, W] in [0, 1]."""
    return x.clamp_min(1e-8).pow(gamma).clamp(0.0, 1.0)


def add_gaussian_noise(x: Tensor, std: float = 0.05) -> Tensor:
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0.0, 1.0)


def lower_contrast(x: Tensor, scale: float = 0.5, offset: float = 0.1) -> Tensor:
    return (x * scale + offset).clamp(0.0, 1.0)


def random_degrade(
    x: Tensor,
    gamma_range: Tuple[float, float] = (0.3, 0.6),
    noise_std_max: float = 0.05,
    contrast_scale_range: Tuple[float, float] = (0.4, 0.9),
    seed: Optional[int] = None,
) -> Tensor:
    """Apply a random combination of darken / noise / contrast drop. Deterministic if `seed` set."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    gamma = float(torch.empty(1).uniform_(*gamma_range, generator=g).item())
    contrast = float(torch.empty(1).uniform_(*contrast_scale_range, generator=g).item())
    y = lower_gamma(x, gamma=gamma)
    y = lower_contrast(y, scale=contrast, offset=0.0)
    if noise_std_max > 0.0:
        std = float(torch.empty(1).uniform_(0.0, noise_std_max, generator=g).item())
        y = add_gaussian_noise(y, std=std)
    return y
