"""Continuous per-pixel tone-mapping actions.

Policy emits raw 3-channel means plus 3-channel log sigmas. We sample in raw
(unconstrained) space, squash to valid action ranges with tanh+affine, and
return both the applied action components and the *raw* sample (needed by PPO
to re-score under the updated policy).

Intentional simplification: log-probabilities are computed in the raw Gaussian
space, ignoring the deterministic tanh+affine squash Jacobian. That Jacobian is
identical for old and new policies on the same raw sample, so it cancels in
the PPO ratio; omitting it simplifies the code without affecting the update.
"""

from dataclasses import dataclass
from typing import Tuple

import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionBounds:
    """Closed intervals for gamma, alpha, beta."""

    gamma: Tuple[float, float]
    alpha: Tuple[float, float]
    beta: Tuple[float, float]

    @staticmethod
    def from_config(action_cfg) -> "ActionBounds":
        return ActionBounds(
            gamma=tuple(action_cfg.gamma_range),
            alpha=tuple(action_cfg.alpha_range),
            beta=tuple(action_cfg.beta_range),
        )


def _affine_from_tanh(t: Tensor, lo: float, hi: float) -> Tensor:
    """Map t in (-1, 1) to (lo, hi)."""
    return lo + (hi - lo) * 0.5 * (t + 1.0)


def raw_to_curve_params(
    raw: Tensor, bounds: ActionBounds
) -> Tuple[Tensor, Tensor, Tensor]:
    """Squash a raw `[B, 3, H, W]` tensor into valid (gamma, alpha, beta) ranges.

    Each output is `[B, 1, H, W]`.
    """
    assert raw.ndim == 4 and raw.shape[1] == 3, f"expected [B,3,H,W], got {tuple(raw.shape)}"
    t = torch.tanh(raw)
    gamma = _affine_from_tanh(t[:, 0:1], *bounds.gamma)
    alpha = _affine_from_tanh(t[:, 1:2], *bounds.alpha)
    beta = _affine_from_tanh(t[:, 2:3], *bounds.beta)
    return gamma, alpha, beta


def _log_prob_raw(raw_sample: Tensor, mu: Tensor, log_sigma: Tensor) -> Tensor:
    """Per-pixel Gaussian log-prob summed across the 3 action channels.

    Returns `[B, 1, H, W]`.
    """
    inv_var = torch.exp(-2.0 * log_sigma)
    sq = (raw_sample - mu) ** 2
    log_prob_per_ch = -0.5 * sq * inv_var - log_sigma - 0.5 * math.log(2.0 * math.pi)
    return log_prob_per_ch.sum(dim=1, keepdim=True)


def sample_action(
    mu: Tensor, log_sigma: Tensor, bounds: ActionBounds
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sample action, return (gamma, alpha, beta, log_prob, raw_sample).

    gamma/alpha/beta are `[B, 1, H, W]`.
    log_prob is `[B, 1, H, W]` (summed over the 3 action channels).
    raw_sample is `[B, 3, H, W]` — must be cached for PPO re-scoring.
    """
    assert mu.shape == log_sigma.shape, f"mu {mu.shape} vs log_sigma {log_sigma.shape}"
    assert mu.ndim == 4 and mu.shape[1] == 3
    sigma = log_sigma.exp()
    eps = torch.randn_like(mu)
    raw_sample = mu + sigma * eps
    log_prob = _log_prob_raw(raw_sample, mu, log_sigma)
    gamma, alpha, beta = raw_to_curve_params(raw_sample, bounds)
    return gamma, alpha, beta, log_prob, raw_sample


def evaluate_log_prob(
    raw_sample: Tensor, mu: Tensor, log_sigma: Tensor
) -> Tensor:
    """Re-score a cached raw sample under a (possibly updated) policy."""
    assert raw_sample.shape == mu.shape == log_sigma.shape
    return _log_prob_raw(raw_sample, mu, log_sigma)


def gaussian_entropy(log_sigma: Tensor) -> Tensor:
    """Per-pixel Gaussian entropy summed over the 3 action channels. Returns `[B, 1, H, W]`."""
    # H = 0.5 * log(2 pi e) + log sigma, per channel. Sum over channels.
    const = 0.5 * math.log(2.0 * math.pi * math.e)
    return (log_sigma + const).sum(dim=1, keepdim=True)


def apply_curve(x: Tensor, gamma: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """Apply per-pixel tone-mapping curve `clip(alpha * x^gamma + beta, 0, 1)`.

    gamma/alpha/beta may have channel dim 1 (broadcast across x's channels).
    """
    assert x.ndim == 4, f"expected 4D, got {tuple(x.shape)}"
    x_safe = x.clamp_min(1e-8)  # avoid 0**gamma NaN gradient
    y = alpha * x_safe.pow(gamma) + beta
    return y.clamp(0.0, 1.0)
