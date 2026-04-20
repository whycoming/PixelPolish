"""Episode rollout collection.

Handles both scalar-reward (Phase 3) and pixel-reward (Phase 4) modes. The
trajectory stores raw samples (not squashed actions) so PPO can re-score them
under the updated policy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from src.env.image_env import ImageEnhancementEnv
from src.models.actions import apply_curve, raw_to_curve_params, sample_action
from src.models.policy_fcn import PolicyValueFCN


@dataclass
class Trajectory:
    """All tensors have a leading time dim T=episode_length.

    states:       [T, B, C, H, W]
    raw_actions:  [T, B, 3, H, W]
    log_probs:    [T, B, 1, H, W]
    rewards:      [T, B]          if reward.mode == 'scalar'
                  [T, B, 1, H, W] if reward.mode == 'pixel'
    values:       [T+1, B, 1, H, W]      bootstrap appended
    sub_rewards:  dict name -> [T] mean-across-batch scalars (for logging)
    final_state:  [B, C, H, W] image after applying all T actions (x_K).
                  Populated by both rollout modes; used for terminal-reward GRPO.
    """

    states: Tensor
    raw_actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    values: Tensor
    sub_rewards: Dict[str, List[float]]
    final_state: Optional[Tensor] = None


def collect_rollout(
    env: ImageEnhancementEnv,
    policy: PolicyValueFCN,
    x0: Tensor,
    reward_mode: str,
) -> Trajectory:
    """Run one full episode from `x0` under the current policy (no gradients)."""
    assert reward_mode in {"scalar", "pixel"}
    device = x0.device
    T = env.episode_length
    B, C, H, W = x0.shape

    states = torch.empty((T, B, C, H, W), dtype=x0.dtype, device=device)
    raw_actions = torch.empty((T, B, 3, H, W), dtype=x0.dtype, device=device)
    log_probs = torch.empty((T, B, 1, H, W), dtype=x0.dtype, device=device)
    values = torch.empty((T + 1, B, 1, H, W), dtype=x0.dtype, device=device)

    if reward_mode == "scalar":
        rewards = torch.empty((T, B), dtype=x0.dtype, device=device)
    else:
        rewards = torch.empty((T, B, 1, H, W), dtype=x0.dtype, device=device)

    sub_rewards: Dict[str, List[float]] = {}

    x_t = env.reset(x0)
    policy.eval()
    with torch.no_grad():
        for t in range(T):
            states[t] = x_t
            mu, log_sigma, v = policy(x_t)
            gamma, alpha, beta, log_prob, raw = sample_action(mu, log_sigma, env.bounds)
            raw_actions[t] = raw
            log_probs[t] = log_prob
            values[t] = v
            x_next, r, _done, info = env.step(gamma, alpha, beta)
            rewards[t] = r
            for k, val in info["sub_rewards"].items():
                sub_rewards.setdefault(k, []).append(float(val))
            x_t = x_next
        # Bootstrap value for GAE.
        _, _, v_last = policy(x_t)
        values[T] = v_last

    return Trajectory(
        states=states,
        raw_actions=raw_actions,
        log_probs=log_probs,
        rewards=rewards,
        values=values,
        sub_rewards=sub_rewards,
        final_state=x_t,
    )


def collect_rollout_actions_only(
    env: ImageEnhancementEnv,
    policy: PolicyValueFCN,
    x0: Tensor,
    shared_global_noise: bool = False,
    aug_channels: Optional[Tensor] = None,
) -> Trajectory:
    """Roll out an episode without computing per-step rewards.

    Avoids the per-step CompositeReward (and its IQA model calls) entirely.
    Used by GRPO in terminal-reward mode where the reward is computed only
    once per rollout from (x_0, x_K). Stores zero placeholders for rewards
    and values so the Trajectory dataclass shape matches the reward-using
    path.

    If `aug_channels` (shape [B, K, H, W]) is provided, they are concatenated
    to x_t along the channel dim before each policy call and stored in
    `states` — this is v6 state augmentation (IQA(x_0) broadcast). The raw
    3-channel x_t is still used for `apply_curve` and for `final_state`.
    """
    device = x0.device
    T = env.episode_length
    B, C, H, W = x0.shape
    K = aug_channels.shape[1] if aug_channels is not None else 0

    states = torch.empty((T, B, C + K, H, W), dtype=x0.dtype, device=device)
    raw_actions = torch.empty((T, B, 3, H, W), dtype=x0.dtype, device=device)
    log_probs = torch.empty((T, B, 1, H, W), dtype=x0.dtype, device=device)

    x_t = x0
    policy.eval()
    with torch.no_grad():
        for t in range(T):
            pin = x_t if aug_channels is None else torch.cat([x_t, aug_channels], dim=1)
            states[t] = pin
            mu, log_sigma, _ = policy(pin)
            gamma, alpha, beta, log_prob, raw = sample_action(
                mu, log_sigma, env.bounds, shared_global_noise=shared_global_noise
            )
            raw_actions[t] = raw
            log_probs[t] = log_prob
            x_t = apply_curve(x_t, gamma, alpha, beta)

    rewards = torch.zeros((T, B), dtype=x0.dtype, device=device)
    values = torch.zeros((T + 1, B, 1, H, W), dtype=x0.dtype, device=device)
    return Trajectory(
        states=states,
        raw_actions=raw_actions,
        log_probs=log_probs,
        rewards=rewards,
        values=values,
        sub_rewards={},
        final_state=x_t,
    )
