"""Generalized Advantage Estimation (Schulman et al. 2016).

Supports both scalar rewards `[T, B]` (with scalar values reduced from
`[T+1, B, 1, H, W]`) and per-pixel rewards `[T, B, 1, H, W]` (with per-pixel
values unchanged). Returns both advantages and returns with the same shape
as the input reward tensor.
"""

from typing import Tuple

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """Compute advantages and returns via GAE.

    Args:
        rewards: `[T, ...]` (any trailing shape compatible with values[:-1]).
        values:  `[T+1, ...]` bootstrapped values; last entry is V(s_T).
        gamma:   discount.
        lam:     GAE lambda.

    Returns:
        (advantages, returns) each with shape `[T, ...]` (= reward shape).
    """
    assert values.shape[0] == rewards.shape[0] + 1
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def reduce_values_to_scalar(values_pix: Tensor) -> Tensor:
    """Reduce per-pixel value `[..., 1, H, W]` to scalar `[...]` by spatial mean."""
    return values_pix.mean(dim=(-1, -2, -3))
