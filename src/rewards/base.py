"""Reward interface.

Convention: `compute(x_prev, x_curr)` returns a tensor that is *higher is
better*, shaped either `[B]` (scalar per image) or `[B, 1, H, W]` (per-pixel).
Whether a reward is spatial is indicated by `spatial` attribute. The
`CompositeReward` handles broadcasting between modes.

All built-in rewards are evaluated with `torch.no_grad()` internally — they
are *signals* for PPO, not differentiable losses.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class RewardFunction(ABC):
    """Abstract reward. Subclasses set `spatial` and override `_compute`."""

    spatial: bool = False

    @abstractmethod
    def _compute(self, x: Tensor) -> Tensor:
        """Compute the reward *value* (not delta) for a single image batch."""

    def compute(self, x_prev: Optional[Tensor], x_curr: Tensor) -> Tensor:
        """Compute absolute reward value on `x_curr`. Higher is better.

        `x_prev` is accepted for API symmetry but ignored here; the relative
        wrapper uses it. Wrap with `torch.no_grad`.
        """
        with torch.no_grad():
            return self._compute(x_curr)

    def to(self, device) -> "RewardFunction":
        """No-op for stateless rewards. Override in IQA wrappers."""
        return self


class RelativeReward:
    """Wraps any RewardFunction to return `R(x_curr) - R(x_prev)`.

    This is the stability trick from CLAUDE.md: absolute IQA scores saturate;
    deltas do not.
    """

    def __init__(self, inner: RewardFunction) -> None:
        self.inner = inner
        self.spatial = inner.spatial

    def compute(self, x_prev: Tensor, x_curr: Tensor) -> Tensor:
        with torch.no_grad():
            return self.inner._compute(x_curr) - self.inner._compute(x_prev)

    def to(self, device) -> "RelativeReward":
        self.inner = self.inner.to(device)
        return self
