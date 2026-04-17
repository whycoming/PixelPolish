"""GRPO trainer (Phase 6 ablation).

For each input image we roll out G independent trajectories; advantages are
computed by group z-normalizing the per-trajectory returns. Critic is
optional — with `drop_critic=True` we skip value regression and use returns
as-is.

Memory scales as B * G * T forwards per update. For B=2, G=8, T=5 on a 4090,
this is 80 forward passes per minibatch — use `minibatch_size` carefully.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from src.algorithms.rollout import Trajectory, collect_rollout
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import evaluate_log_prob, gaussian_entropy
from src.models.policy_fcn import PolicyValueFCN


@dataclass
class GRPOConfig:
    group_size: int = 8
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_min: float = 0.001
    entropy_anneal_steps: int = 20000
    ppo_epochs: int = 4
    minibatch_size: int = 8
    max_grad_norm: float = 0.5
    drop_critic: bool = True
    beta_kl: float = 0.0
    reward_mode: str = "scalar"


def _trajectory_return_scalar(traj: Trajectory) -> Tensor:
    """Sum of per-step rewards, reduced to [B] regardless of reward mode."""
    r = traj.rewards
    if r.ndim == 2:
        return r.sum(dim=0)  # [B]
    # [T, B, 1, H, W] -> sum over T, mean over spatial -> [B]
    return r.sum(dim=0).mean(dim=(1, 2, 3))


class GRPOTrainer:
    """Group-normalized PPO-style trainer."""

    def __init__(
        self,
        policy: PolicyValueFCN,
        optimizer: Optimizer,
        env: ImageEnhancementEnv,
        cfg: GRPOConfig,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.env = env
        self.cfg = cfg
        self._update_step = 0

    def _current_entropy_coef(self) -> float:
        c = self.cfg
        if c.entropy_anneal_steps <= 0:
            return c.entropy_coef
        frac = min(1.0, self._update_step / float(c.entropy_anneal_steps))
        return c.entropy_coef + (c.entropy_coef_min - c.entropy_coef) * frac

    def _collect_group(self, x0: Tensor) -> List[Trajectory]:
        """Roll out `group_size` trajectories from the same x0."""
        return [collect_rollout(self.env, self.policy, x0, self.cfg.reward_mode)
                for _ in range(self.cfg.group_size)]

    def update(self, x0: Tensor) -> Dict[str, float]:
        """Collect G rollouts from `x0` and perform PPO-style update using group advantage."""
        cfg = self.cfg
        device = x0.device
        trajs = self._collect_group(x0)

        # Per-trajectory scalar return -> [G, B]
        returns_g = torch.stack([_trajectory_return_scalar(t) for t in trajs], dim=0)
        # Group z-normalize across G for each image in batch.
        mean_g = returns_g.mean(dim=0, keepdim=True)
        std_g = returns_g.std(dim=0, keepdim=True).clamp_min(1e-6)
        adv_g = (returns_g - mean_g) / std_g  # [G, B]

        # Flatten G and T into a single leading dim for minibatch sampling.
        # Shapes we need per sample:
        #   states:      [C, H, W]
        #   raw_actions: [3, H, W]
        #   log_probs:   [1, H, W]
        #   adv:         scalar (broadcast to pixel when used)
        # We build lists then stack.
        s_list, a_list, lp_list, adv_list, ret_list = [], [], [], [], []
        for g, traj in enumerate(trajs):
            T_, B_ = traj.states.shape[0], traj.states.shape[1]
            s = traj.states.reshape(T_ * B_, *traj.states.shape[2:])
            a = traj.raw_actions.reshape(T_ * B_, *traj.raw_actions.shape[2:])
            lp = traj.log_probs.reshape(T_ * B_, *traj.log_probs.shape[2:])
            # Broadcast adv_g[g, b] to each timestep.
            adv_gb = adv_g[g].unsqueeze(0).expand(T_, -1).reshape(T_ * B_)
            ret_gb = returns_g[g].unsqueeze(0).expand(T_, -1).reshape(T_ * B_)
            s_list.append(s); a_list.append(a); lp_list.append(lp)
            adv_list.append(adv_gb); ret_list.append(ret_gb)

        states = torch.cat(s_list, dim=0)
        raw_actions = torch.cat(a_list, dim=0)
        log_probs_old = torch.cat(lp_list, dim=0)
        advantages = torch.cat(adv_list, dim=0)
        returns = torch.cat(ret_list, dim=0)

        N = states.shape[0]
        mb = max(1, cfg.minibatch_size)
        ent_coef = self._current_entropy_coef()

        metrics: Dict[str, float] = {
            "loss/total": 0.0, "loss/policy": 0.0, "loss/value": 0.0,
            "loss/entropy": 0.0, "policy/kl_approx": 0.0, "policy/clip_frac": 0.0,
            "grpo/return_mean": float(returns_g.mean()),
            "grpo/return_std": float(returns_g.std()),
        }
        n = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, mb):
                sel = perm[start:start + mb]
                s = states[sel]
                a = raw_actions[sel]
                lp_old = log_probs_old[sel]
                adv = advantages[sel]
                ret = returns[sel]

                mu, log_sigma, v_pix = self.policy(s)
                lp_new = evaluate_log_prob(a, mu, log_sigma)
                entropy = gaussian_entropy(log_sigma)

                lp_new_s = lp_new.mean(dim=(1, 2, 3))
                lp_old_s = lp_old.mean(dim=(1, 2, 3))
                v_s = v_pix.mean(dim=(1, 2, 3))

                ratio = torch.exp(lp_new_s - lp_old_s)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.tensor(0.0, device=device)
                if not cfg.drop_critic:
                    value_loss = F.mse_loss(v_s, ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (lp_old_s - lp_new_s).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                    metrics["loss/total"] += float(loss.detach())
                    metrics["loss/policy"] += float(policy_loss.detach())
                    metrics["loss/value"] += float(value_loss.detach())
                    metrics["loss/entropy"] += float(entropy_loss.detach())
                    metrics["policy/kl_approx"] += float(approx_kl)
                    metrics["policy/clip_frac"] += float(clip_frac)
                n += 1

        for k in ("loss/total", "loss/policy", "loss/value", "loss/entropy",
                  "policy/kl_approx", "policy/clip_frac"):
            metrics[k] /= max(1, n)
        metrics["policy/entropy_coef"] = ent_coef
        self._update_step += 1
        return metrics
