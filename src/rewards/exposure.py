"""Zero-DCE-style local exposure anchor for terminal-Borda.

Computes per-image score `-mean_patch(|patch_mean(luma) - target|²)`. Higher
(closer to 0) means the image's patch-wise luma is closer to `target` on
average. Bidirectional: penalises both under- and over-exposure equally, so
unlike CLIP-IQA / MUSIQ it cannot reward unidirectional brightening.

Used as a third Borda head alongside clipiqa/musiq to prevent the "more is
more" failure mode (Zero-DCE Guo et al., CVPR 2020 — L_exposure).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from src.rewards.base import RewardFunction


_LUMA = (0.299, 0.587, 0.114)


def _to_luma(x: Tensor) -> Tensor:
    """[B,C,H,W] -> [B,1,H,W] luma. C=1 passes through; C=3 uses BT.601 weights."""
    assert x.ndim == 4, f"expected [B,C,H,W], got {tuple(x.shape)}"
    if x.shape[1] == 1:
        return x
    if x.shape[1] == 3:
        r, g, b = _LUMA
        return (r * x[:, 0:1] + g * x[:, 1:2] + b * x[:, 2:3])
    return x.mean(dim=1, keepdim=True)


class LocalExposureReward(RewardFunction):
    """`-mean(|patch_mean(luma) - target|²)` per image. Higher = better."""

    spatial = False

    def __init__(self, patch_size: int = 16, target: float = 0.6,
                 device: str = "cpu") -> None:
        self.patch_size = int(patch_size)
        self.target = float(target)
        self._device = str(device)

    def to(self, device) -> "LocalExposureReward":
        self._device = str(device)
        return self

    def _compute(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            y = _to_luma(x)
            patch_mean = F.avg_pool2d(y, kernel_size=self.patch_size,
                                      stride=self.patch_size, ceil_mode=True)
            err = (patch_mean - self.target).pow(2)
            score = -err.mean(dim=(1, 2, 3))  # [B]
        return score.to(x.device).float().view(-1)
