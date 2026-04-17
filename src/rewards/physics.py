"""Physics-based no-reference rewards: gradient magnitude, entropy, EME.

All implementations are pure PyTorch (no numpy detours), accept `[B, C, H, W]`
images in `[0, 1]`, and return either `[B]` or `[B, 1, H, W]` depending on
their `spatial` attribute. Higher is better.
"""

from typing import Optional

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.rewards.base import RewardFunction


def _to_gray(x: Tensor) -> Tensor:
    """Rec. 601 luminance for [B, C, H, W] with C in {1, 3}. Returns [B, 1, H, W]."""
    if x.shape[1] == 1:
        return x
    if x.shape[1] == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    # Arbitrary channel count: take mean.
    return x.mean(dim=1, keepdim=True)


class GradientReward(RewardFunction):
    """Mean Sobel gradient magnitude. Encourages edge contrast."""

    spatial = True  # per-pixel magnitude; reduced to scalar only in composite

    def __init__(self) -> None:
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = kx.t().contiguous()
        self._kx = kx.view(1, 1, 3, 3)
        self._ky = ky.view(1, 1, 3, 3)

    def _compute(self, x: Tensor) -> Tensor:
        gray = _to_gray(x)
        kx = self._kx.to(gray.device, gray.dtype)
        ky = self._ky.to(gray.device, gray.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)  # [B, 1, H, W]


class EntropyReward(RewardFunction):
    """Normalized image entropy via soft histogram binning (Shannon).

    Soft binning lets this play nicely if ever used as a loss, and is smooth.
    """

    spatial = False

    def __init__(self, num_bins: int = 64, sigma: float = 0.02) -> None:
        self.num_bins = int(num_bins)
        self.sigma = float(sigma)
        centers = torch.linspace(0.0, 1.0, num_bins)
        self._centers = centers.view(1, num_bins, 1)  # [1, K, 1]

    def _compute(self, x: Tensor) -> Tensor:
        gray = _to_gray(x).flatten(start_dim=1)  # [B, HW]
        centers = self._centers.to(gray.device, gray.dtype)  # [1, K, 1]
        diff = gray.unsqueeze(1) - centers  # [B, K, HW]
        soft = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        soft = soft / (soft.sum(dim=1, keepdim=True) + 1e-8)
        hist = soft.sum(dim=2)  # [B, K]
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(hist * torch.log(hist + 1e-8)).sum(dim=1)  # [B]
        # Normalize by log(K) to [0, 1] range.
        return entropy / math.log(self.num_bins)


class EMEReward(RewardFunction):
    """Agaian's Measure of Enhancement by Entropy (EME), block-based.

    EME = mean over non-overlapping blocks of `20 * log10(max/min)`. Higher =
    more local contrast. We take gray, block max/min via average-pooled max and
    min, and log-ratio.
    """

    spatial = False

    def __init__(self, block: int = 8) -> None:
        self.block = int(block)

    def _compute(self, x: Tensor) -> Tensor:
        gray = _to_gray(x)  # [B, 1, H, W]
        b = self.block
        # Crop to multiples of block for clean tiling.
        _, _, h, w = gray.shape
        h2, w2 = (h // b) * b, (w // b) * b
        if h2 == 0 or w2 == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        gray = gray[..., :h2, :w2]
        # Block min = -max(-x). Use max_pool2d for both.
        block_max = F.max_pool2d(gray, kernel_size=b, stride=b)
        block_min = -F.max_pool2d(-gray, kernel_size=b, stride=b)
        eps = 1e-4
        ratio = (block_max + eps) / (block_min + eps)
        eme_per_block = 20.0 * torch.log10(ratio)
        return eme_per_block.mean(dim=(1, 2, 3))  # [B]
