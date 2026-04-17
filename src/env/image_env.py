"""Gym-style per-pixel image enhancement environment.

Unlike Gym envs that wrap a single observation, this operates on an entire
batch: `reset(x0)` loads `[B, C, H, W]` images, `step(action)` applies the
curve to every image in the batch and returns the new image plus per-image or
per-pixel reward.

Episodes are fixed-length (K steps from config). No learned stopping.
"""

from typing import Tuple

from torch import Tensor

from src.models.actions import ActionBounds, apply_curve
from src.rewards.composite import CompositeReward


class ImageEnhancementEnv:
    """Batched image-enhancement env with fixed-length episodes."""

    def __init__(
        self,
        reward_fn: CompositeReward,
        bounds: ActionBounds,
        episode_length: int = 5,
    ) -> None:
        self.reward_fn = reward_fn
        self.bounds = bounds
        self.episode_length = int(episode_length)
        self._x: Tensor | None = None
        self._t: int = 0

    def reset(self, x0: Tensor) -> Tensor:
        """Load a fresh batch. Returns `x0` unchanged for symmetry with Gym APIs."""
        assert x0.ndim == 4
        self._x = x0
        self._t = 0
        return self._x

    def step(
        self, gamma: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tuple[Tensor, Tensor, bool, dict]:
        """Apply action, compute reward, advance timestep.

        Returns (next_image, reward, done, info). Reward shape depends on the
        composite's mode (`scalar` -> `[B]`, `pixel` -> `[B, 1, H, W]`).
        """
        assert self._x is not None, "call reset(x0) before step()."
        x_prev = self._x
        x_next = apply_curve(x_prev, gamma, alpha, beta)
        reward, per_name = self.reward_fn.compute(x_prev, x_next)
        self._x = x_next
        self._t += 1
        done = self._t >= self.episode_length
        return x_next, reward, done, {"sub_rewards": per_name}
