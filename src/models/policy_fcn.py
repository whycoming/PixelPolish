"""Shared FCN backbone with policy (mu, log_sigma) and value heads.

Architecture follows PixelRL (Furuta et al. 2020): stacked dilated 3x3
convolutions (dilations 1, 2, 3, 4) to grow receptive field without
downsampling. Output resolution matches input.
"""

from typing import Tuple

import torch
from torch import Tensor, nn


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


_LOG_SIGMA_MIN = -5.0
_LOG_SIGMA_MAX = 2.0


class PolicyValueFCN(nn.Module):
    """Fully-convolutional shared trunk with (mu, log_sigma) policy and value heads.

    Forward returns (mu, log_sigma, value). Shapes:
      mu:        `[B, 3, H, W]` raw means for (gamma, alpha, beta).
      log_sigma: `[B, 3, H, W]` clamped to [_LOG_SIGMA_MIN, _LOG_SIGMA_MAX].
      value:     `[B, 1, H, W]` per-pixel value estimate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        num_dilated_blocks: int = 4,
        init_log_sigma: float = -0.5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[_DilatedBlock(base_filters, d) for d in range(1, num_dilated_blocks + 1)]
        )
        # Extra refinement conv before heads, standard PixelRL trick.
        self.refine = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.policy_mu_head = nn.Conv2d(base_filters, 3, kernel_size=1)
        self.policy_log_sigma_head = nn.Conv2d(base_filters, 3, kernel_size=1)
        self.value_head = nn.Conv2d(base_filters, 1, kernel_size=1)

        nn.init.zeros_(self.policy_mu_head.weight)
        nn.init.zeros_(self.policy_mu_head.bias)
        nn.init.zeros_(self.policy_log_sigma_head.weight)
        nn.init.constant_(self.policy_log_sigma_head.bias, float(init_log_sigma))
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert x.ndim == 4, f"expected [B,C,H,W], got {tuple(x.shape)}"
        h = self.refine(self.trunk(self.stem(x)))
        mu = self.policy_mu_head(h)
        log_sigma = self.policy_log_sigma_head(h).clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        value = self.value_head(h)
        return mu, log_sigma, value
