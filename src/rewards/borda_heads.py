"""Input-relative Borda heads for GRPO terminal-Borda reward (v6).

All heads have unified signature `fn(x_0, x_K) -> [B]` (higher is better).
Input-relative heads use x_0 as an anchor; unary IQA heads ignore it.

Theoretical purpose (Gao et al. 2023 scaling-law bound): a reward R̂ that is
**bounded and input-anchored** gives Goodhart-free scaling under KL
regularization. v4/v5 failed because clipiqa/musiq target-mode is anchored
to the *training distribution*, not the current input — on out-of-distribution
test images the policy learned to regress outputs to the training mean IQA.
These heads fix that: identity_l1 and lpips are bounded above by 0 and reach
0 only at x_K = x_0 (or perceptually identical); gray_world is bounded.
"""

from typing import Optional

import torch
from torch import Tensor

from src.rewards.base import RewardFunction


class IdentityL1Head(RewardFunction):
    """`-mean |x_K - x_0|` per image. Bounded ≤ 0, max (=0) at x_K = x_0."""

    spatial = False

    def __init__(self, device: str = "cpu") -> None:
        self._device = str(device)

    def to(self, device) -> "IdentityL1Head":
        self._device = str(device)
        return self

    def _compute(self, x_0: Tensor, x_K: Tensor) -> Tensor:
        with torch.no_grad():
            d = (x_K - x_0).abs().mean(dim=(1, 2, 3))
        return (-d).to(x_K.device).float().view(-1)


class GrayWorldHead(RewardFunction):
    """Zero-DCE `L_color`: `-(|R̄-Ḡ| + |Ḡ-B̄| + |R̄-B̄|)` per image.

    Encourages channel-mean parity (gray-world prior). Ignores x_0. Bounded
    ≤ 0; equals 0 when the three channel means are equal.
    """

    spatial = False

    def __init__(self, device: str = "cpu") -> None:
        self._device = str(device)

    def to(self, device) -> "GrayWorldHead":
        self._device = str(device)
        return self

    def _compute(self, x_0: Tensor, x_K: Tensor) -> Tensor:
        del x_0  # unused
        with torch.no_grad():
            if x_K.shape[1] != 3:
                return torch.zeros(x_K.shape[0], device=x_K.device, dtype=x_K.dtype)
            means = x_K.mean(dim=(2, 3))  # [B, 3]
            r, g, b = means[:, 0], means[:, 1], means[:, 2]
            err = (r - g).abs() + (g - b).abs() + (r - b).abs()
        return (-err).to(x_K.device).float().view(-1)


class LPIPSHead(RewardFunction):
    """`-LPIPS(x_K, x_0)` per image (VGG16 backbone, frozen). Bounded ≤ 0.

    LPIPS ≥ 0 with LPIPS(x, x) = 0. Penalises perceptual distance from the
    input, including structural and color shifts that pure L1 misses.
    """

    spatial = False

    def __init__(self, net: str = "vgg", device: str = "cpu") -> None:
        import lpips  # type: ignore
        self._lpips = lpips.LPIPS(net=net, verbose=False)
        for p in self._lpips.parameters():
            p.requires_grad_(False)
        self._lpips.eval()
        self._lpips.to(device)
        self._device = str(device)

    def to(self, device) -> "LPIPSHead":
        self._device = str(device)
        self._lpips.to(device)
        return self

    def _compute(self, x_0: Tensor, x_K: Tensor) -> Tensor:
        with torch.no_grad():
            # lpips expects [-1, 1] range
            a = (x_0.to(self._device) * 2.0 - 1.0).float()
            b = (x_K.to(self._device) * 2.0 - 1.0).float()
            # LPIPS only supports 3-channel. Expand grayscale if needed.
            if a.shape[1] == 1:
                a = a.expand(-1, 3, -1, -1)
                b = b.expand(-1, 3, -1, -1)
            d = self._lpips(a, b).view(-1)  # [B]
        return (-d).to(x_K.device).float().view(-1)
