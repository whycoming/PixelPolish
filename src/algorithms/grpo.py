"""GRPO trainer (Phase 6 ablation, plus theory-driven anti-hacking mode).

Two operating modes:

1. *Per-step composite reward* (legacy): each step uses the env's CompositeReward,
   trajectory return is summed over time, advantages are group z-normalized.

2. *Terminal Borda* (recommended; enable via `reward_mode='terminal_borda'`):
   only one reward signal per rollout = ranks against group siblings on K
   independent NR-IQA heads applied to (x_0, x_K). Aggregation = average of
   per-head ranks (Borda count), centered. This avoids any assumption about
   reward distribution shape and is robust to a single head being adversarially
   exploited — the policy must beat its siblings on a *majority* of heads to
   gain advantage. Combined with KL-to-identity (`beta_kl > 0`), this gives
   the Goodhart-bounded scaling-law guarantee from Gao et al. 2023.

KL term is computed in *raw* Gaussian space (before tanh+affine) against a
fixed reference policy with μ_ref=0, log_σ_ref=`init_log_sigma_ref`. With
log-uniform γ/α parameterization centered at 1 and symmetric β, μ_ref=0
corresponds to the *identity* action (γ=1, α=1, β=0).

Memory scales as B * G * T forwards per update.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from src.algorithms.rollout import (
    Trajectory,
    collect_rollout,
    collect_rollout_actions_only,
)
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
    init_log_sigma_ref: float = -0.5
    reward_mode: str = "scalar"  # {scalar, pixel, terminal_borda}
    borda_heads: List[str] = field(default_factory=list)
    shared_global_noise: bool = False  # if True, sibling rollouts differ only by a global tone shift (per-rollout [B,3,1,1] noise broadcast across H,W). Restores rank informativeness for IQA-Borda.
    # Per-head (μ, σ) of natural-image IQA distribution. When set, terminal-Borda
    # scores rollouts by negative squared z-distance from μ instead of Δ from x_0:
    #   score_h(rollout) = -((IQA_h(x_K) - μ_h) / σ_h)²
    # This anchors the reward to "natural-typical IQA" (Option A target reward),
    # eliminating the "more is more" failure mode where heads with intrinsic
    # brightness/contrast bias push already-good images further out of distribution.
    iqa_targets: Optional[Dict[str, Tuple[float, float]]] = None


def _trajectory_return_scalar(traj: Trajectory) -> Tensor:
    """Sum of per-step rewards, reduced to [B] regardless of reward mode."""
    r = traj.rewards
    if r.ndim == 2:
        return r.sum(dim=0)  # [B]
    # [T, B, 1, H, W] -> sum over T, mean over spatial -> [B]
    return r.sum(dim=0).mean(dim=(1, 2, 3))


def _borda_rank_advantage(
    head_scores: Dict[str, Tensor],  # name -> [G, B] absolute scores on x_K (or deltas)
) -> Tensor:
    """Average per-head ranks across IQA heads, centered to mean 0.

    Heads are ranked independently (Borda count) so a single head being
    adversarially exploited by the policy contributes only one vote among
    K. Returned tensor has shape [G, B] and is approximately in the range
    roughly [-(G-1)/2, (G-1)/2] (centered ranks).
    """
    if len(head_scores) == 0:
        raise ValueError("Borda advantage requires at least one IQA head.")
    rank_sum: Optional[Tensor] = None
    for name, s in head_scores.items():
        # `argsort` returns indices that would sort. Apply twice: ranks.
        # We want higher score -> higher rank, so sort descending.
        order = torch.argsort(s, dim=0, descending=True)  # [G, B]
        ranks = torch.empty_like(order, dtype=s.dtype)
        # ranks[order[i, b], b] = G-1-i  (0 = worst, G-1 = best)
        idx = torch.arange(s.shape[0], device=s.device, dtype=s.dtype)
        ranks.scatter_(0, order, (s.shape[0] - 1 - idx).unsqueeze(1).expand_as(order))
        rank_sum = ranks if rank_sum is None else rank_sum + ranks
    avg_rank = rank_sum / float(len(head_scores))  # [G, B]
    # Center: 0-mean across G.
    return avg_rank - avg_rank.mean(dim=0, keepdim=True)


def _kl_to_identity(
    mu: Tensor, log_sigma: Tensor, log_sigma_ref: float
) -> Tensor:
    """KL( N(μ, σ²) || N(0, σ_ref²) ) per pixel, summed over the 3 channels.

    Closed form for diagonal Gaussians (per-channel):
        KL = log(σ_ref/σ) + (σ² + μ²) / (2 σ_ref²) - 0.5
    Identity policy in *raw* space is μ=0; with log-uniform γ/α + symmetric β,
    raw=0 maps deterministically to (γ=1, α=1, β=0) = the identity action.
    """
    inv_var_ref = math.exp(-2.0 * log_sigma_ref)
    sigma2 = (2.0 * log_sigma).exp()
    per_ch = (
        log_sigma_ref - log_sigma
        + 0.5 * (sigma2 + mu ** 2) * inv_var_ref
        - 0.5
    )
    return per_ch.sum(dim=1, keepdim=True)


class GRPOTrainer:
    """Group-normalized PPO-style trainer."""

    def __init__(
        self,
        policy: PolicyValueFCN,
        optimizer: Optimizer,
        env: ImageEnhancementEnv,
        cfg: GRPOConfig,
        iqa_heads: Optional[Dict[str, Callable[[Tensor], Tensor]]] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.env = env
        self.cfg = cfg
        self._update_step = 0
        self._iqa_heads = iqa_heads or {}
        if cfg.reward_mode == "terminal_borda" and not self._iqa_heads:
            raise ValueError(
                "reward_mode='terminal_borda' requires at least one IQA head; "
                "pass `iqa_heads={name: fn(x)->[B]}` to GRPOTrainer."
            )

    def _current_entropy_coef(self) -> float:
        c = self.cfg
        if c.entropy_anneal_steps <= 0:
            return c.entropy_coef
        frac = min(1.0, self._update_step / float(c.entropy_anneal_steps))
        return c.entropy_coef + (c.entropy_coef_min - c.entropy_coef) * frac

    def _collect_group(self, x0: Tensor) -> List[Trajectory]:
        """Roll out `group_size` trajectories from the same x0."""
        if self.cfg.reward_mode == "terminal_borda":
            return [collect_rollout_actions_only(
                        self.env, self.policy, x0,
                        shared_global_noise=self.cfg.shared_global_noise)
                    for _ in range(self.cfg.group_size)]
        return [collect_rollout(self.env, self.policy, x0, self.cfg.reward_mode)
                for _ in range(self.cfg.group_size)]

    def _terminal_borda_advantage(
        self, x0: Tensor, trajs: List[Trajectory]
    ) -> Dict[str, Tensor]:
        """Compute per-head scores, rank them, return centered Borda advantage.

        Two scoring modes:

        - **Δ-from-baseline** (default, when `cfg.iqa_targets` is None):
              score_h(rollout) = IQA_h(x_K) - IQA_h(x_0)
          Higher is better — the rollout that improves IQA most against x_0
          gets the top rank. Suffers from "more is more": already-good images
          get pushed further when heads have intrinsic brightness/contrast bias.

        - **Target-distribution** (Option A, when `cfg.iqa_targets` is set):
              score_h(rollout) = -((IQA_h(x_K) - μ_h) / σ_h)²
          Maximized at IQA(x_K) = μ_h, strictly negative either side. Anchors
          the policy to natural-typical IQA per head, eliminating run-away
          deviation. (μ_h, σ_h) precomputed on a high-quality reference set.

        Returns:
            'adv':            [G, B] centered Borda advantage
            'returns':        [G, B] mean per-head score (logging)
            'per_head_score': dict name -> [G, B] raw per-head scores (logging)
            'per_head_delta': alias kept for backward-compat metric naming
        """
        targets = self.cfg.iqa_targets
        head_scores: Dict[str, Tensor] = {}
        # Also collect raw IQA(x_K) per head for diagnostics regardless of mode.
        head_raw_xk: Dict[str, Tensor] = {}
        with torch.no_grad():
            for name, fn in self._iqa_heads.items():
                # Always compute s0 (cheap, one call) for Δ logging in either mode.
                s0 = fn(x0)
                if s0.ndim != 1:
                    s0 = s0.view(-1)
                sk_list, score_list = [], []
                for traj in trajs:
                    sk = fn(traj.final_state)
                    if sk.ndim != 1:
                        sk = sk.view(-1)
                    sk_list.append(sk)
                    if targets is not None and name in targets:
                        mu, sigma = targets[name]
                        z = (sk - float(mu)) / max(float(sigma), 1e-6)
                        score_list.append(-(z * z))
                    else:
                        # Fallback: Δ-from-baseline if no target for this head.
                        score_list.append(sk - s0)
                head_scores[name] = torch.stack(score_list, dim=0)   # [G, B]
                head_raw_xk[name] = torch.stack(sk_list, dim=0)      # [G, B]
        adv = _borda_rank_advantage(head_scores)
        mean_score = torch.stack(list(head_scores.values()), dim=0).mean(dim=0)  # [G, B]
        return {
            "adv": adv,
            "returns": mean_score,
            "per_head_score": head_scores,
            "per_head_delta": head_scores,  # back-compat alias for metric loop
            "per_head_xk": head_raw_xk,
        }

    def update(self, x0: Tensor) -> Dict[str, float]:
        """Collect G rollouts from `x0` and perform PPO-style update using group advantage."""
        cfg = self.cfg
        device = x0.device
        trajs = self._collect_group(x0)

        if cfg.reward_mode == "terminal_borda":
            borda = self._terminal_borda_advantage(x0, trajs)
            adv_g = borda["adv"]            # [G, B]
            returns_g = borda["returns"]    # [G, B], for logging / value head if any
        else:
            # Per-trajectory scalar return -> [G, B]
            returns_g = torch.stack([_trajectory_return_scalar(t) for t in trajs], dim=0)
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
            "loss/entropy": 0.0, "loss/kl_ref": 0.0,
            "policy/kl_approx": 0.0, "policy/clip_frac": 0.0,
            "grpo/return_mean": float(returns_g.mean()),
            "grpo/return_std": float(returns_g.std()),
        }
        if cfg.reward_mode == "terminal_borda":
            for name, score in borda["per_head_score"].items():
                metrics[f"borda/score_{name}"] = float(score.mean())
            for name, xk in borda.get("per_head_xk", {}).items():
                metrics[f"borda/xk_{name}_mean"] = float(xk.mean())
                metrics[f"borda/xk_{name}_std"] = float(xk.std())
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

                kl_ref = torch.tensor(0.0, device=device)
                if cfg.beta_kl > 0.0:
                    kl_ref = _kl_to_identity(mu, log_sigma, cfg.init_log_sigma_ref).mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + ent_coef * entropy_loss
                    + cfg.beta_kl * kl_ref
                )

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
                    metrics["loss/kl_ref"] += float(kl_ref.detach())
                    metrics["policy/kl_approx"] += float(approx_kl)
                    metrics["policy/clip_frac"] += float(clip_frac)
                n += 1

        for k in ("loss/total", "loss/policy", "loss/value", "loss/entropy",
                  "loss/kl_ref", "policy/kl_approx", "policy/clip_frac"):
            metrics[k] /= max(1, n)
        metrics["policy/entropy_coef"] = ent_coef
        self._update_step += 1
        return metrics
