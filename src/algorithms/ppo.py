"""PPO trainer supporting both scalar (Phase 3) and per-pixel (Phase 4) rewards.

Shapes used throughout:
  states:      [T, B, C, H, W]
  raw_actions: [T, B, 3, H, W]
  log_probs:   [T, B, 1, H, W]       (sum over 3 action channels)
  values_pix:  [T+1, B, 1, H, W]
  rewards:     [T, B]                (scalar mode)   OR [T, B, 1, H, W] (pixel mode)
  advantages:  same shape as rewards
  returns:     same shape as rewards

In scalar mode we reduce per-pixel values/log_probs to per-image via spatial
mean before computing the PPO ratio; in pixel mode we keep everything
per-pixel so each pixel has its own advantage.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from src.algorithms.advantage import compute_gae, reduce_values_to_scalar
from src.algorithms.rollout import Trajectory
from src.models.actions import evaluate_log_prob, gaussian_entropy
from src.models.policy_fcn import PolicyValueFCN


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_min: float = 0.001
    entropy_anneal_steps: int = 20000
    ppo_epochs: int = 4
    minibatch_size: int = 8
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    reward_mode: str = "scalar"


class PPOTrainer:
    """Runs PPO updates on collected trajectories. Stateless across updates."""

    def __init__(
        self,
        policy: PolicyValueFCN,
        optimizer: Optimizer,
        cfg: PPOConfig,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.cfg = cfg
        self._update_step = 0

    def _current_entropy_coef(self) -> float:
        c = self.cfg
        if c.entropy_anneal_steps <= 0:
            return c.entropy_coef
        frac = min(1.0, self._update_step / float(c.entropy_anneal_steps))
        return c.entropy_coef + (c.entropy_coef_min - c.entropy_coef) * frac

    def update(self, traj: Trajectory) -> Dict[str, float]:
        """Run PPO epochs on one trajectory. Returns scalar metrics for logging."""
        cfg = self.cfg
        device = traj.states.device

        # ---- Compute advantages ----
        if cfg.reward_mode == "scalar":
            # reduce per-pixel values to scalar per-image
            values = reduce_values_to_scalar(traj.values)  # [T+1, B]
            rewards = traj.rewards  # [T, B]
            advantages, returns = compute_gae(
                rewards, values, gamma=cfg.gae_gamma, lam=cfg.gae_lambda
            )  # [T, B], [T, B]
        else:
            # keep per-pixel values and rewards
            values = traj.values  # [T+1, B, 1, H, W]
            rewards = traj.rewards  # [T, B, 1, H, W]
            advantages, returns = compute_gae(
                rewards, values, gamma=cfg.gae_gamma, lam=cfg.gae_lambda
            )  # [T, B, 1, H, W] each

        if cfg.normalize_advantage:
            adv_flat = advantages.flatten()
            advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # ---- Flatten time and batch for minibatch sampling ----
        T, B = traj.states.shape[0], traj.states.shape[1]
        states = traj.states.reshape(T * B, *traj.states.shape[2:])
        raw_actions = traj.raw_actions.reshape(T * B, *traj.raw_actions.shape[2:])
        log_probs_old = traj.log_probs.reshape(T * B, *traj.log_probs.shape[2:])

        if cfg.reward_mode == "scalar":
            advantages_flat = advantages.reshape(T * B)
            returns_flat = returns.reshape(T * B)
        else:
            advantages_flat = advantages.reshape(T * B, *advantages.shape[2:])  # [N, 1, H, W]
            returns_flat = returns.reshape(T * B, *returns.shape[2:])

        N = T * B
        idx = torch.arange(N, device=device)

        # ---- Accumulators ----
        metrics: Dict[str, float] = {
            "loss/total": 0.0, "loss/policy": 0.0, "loss/value": 0.0,
            "loss/entropy": 0.0, "policy/kl_approx": 0.0, "policy/clip_frac": 0.0,
            "policy/ratio_mean": 0.0,
        }
        ent_coef = self._current_entropy_coef()
        mb = max(1, cfg.minibatch_size)
        n_batches = 0

        for _ in range(cfg.ppo_epochs):
            perm = idx[torch.randperm(N, device=device)]
            for start in range(0, N, mb):
                sel = perm[start:start + mb]
                s = states[sel]
                a = raw_actions[sel]
                lp_old = log_probs_old[sel]

                mu, log_sigma, v_pix = self.policy(s)
                lp_new = evaluate_log_prob(a, mu, log_sigma)  # [b, 1, H, W]
                entropy = gaussian_entropy(log_sigma)  # [b, 1, H, W]

                if cfg.reward_mode == "scalar":
                    # Reduce per-pixel log-prob to per-image sum over spatial dim.
                    # PPO ratio operates per-image: ratio = exp(sum(log_prob))
                    # To keep scale sensible, we mean-reduce instead of sum; this
                    # is an action-averaged policy-gradient proxy, equivalent up
                    # to a constant multiplier that is absorbed in the ratio cap.
                    lp_new_s = lp_new.mean(dim=(1, 2, 3))  # [b]
                    lp_old_s = lp_old.mean(dim=(1, 2, 3))
                    adv_s = advantages_flat[sel]  # [b]
                    ret_s = returns_flat[sel]  # [b]
                    v_s = v_pix.mean(dim=(1, 2, 3))  # [b]
                    ent_s = entropy.mean(dim=(1, 2, 3))  # [b]

                    ratio = torch.exp(lp_new_s - lp_old_s)
                    unclipped = ratio * adv_s
                    clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_s
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(v_s, ret_s)
                    entropy_loss = -ent_s.mean()
                else:
                    # pixel mode: ratio is per-pixel, loss averages over pixels + batch.
                    adv_p = advantages_flat[sel]  # [b, 1, H, W]
                    ret_p = returns_flat[sel]     # [b, 1, H, W]
                    ratio = torch.exp(lp_new - lp_old)
                    unclipped = ratio * adv_p
                    clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_p
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(v_pix, ret_p)
                    entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (lp_old - lp_new).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                    metrics["loss/total"] += float(loss.detach())
                    metrics["loss/policy"] += float(policy_loss.detach())
                    metrics["loss/value"] += float(value_loss.detach())
                    metrics["loss/entropy"] += float(entropy_loss.detach())
                    metrics["policy/kl_approx"] += float(approx_kl)
                    metrics["policy/clip_frac"] += float(clip_frac)
                    metrics["policy/ratio_mean"] += float(ratio.mean())
                n_batches += 1

        for k in metrics:
            metrics[k] /= max(1, n_batches)
        metrics["policy/entropy_coef"] = ent_coef
        self._update_step += 1
        return metrics
