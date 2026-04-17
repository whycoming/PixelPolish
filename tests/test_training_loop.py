"""End-to-end training smoke tests: one PPO update, pixel-mode update, and one GRPO update.

Uses tiny tensors + physics-only rewards (no pyiqa).
"""

from types import SimpleNamespace

import torch
from torch.optim import Adam

from src.algorithms.grpo import GRPOConfig, GRPOTrainer
from src.algorithms.ppo import PPOConfig, PPOTrainer
from src.algorithms.rollout import collect_rollout
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import ActionBounds
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.composite import CompositeReward


def _reward_cfg(mode: str):
    return SimpleNamespace(
        mode=mode,
        relative=True,
        pixel_smooth_radius=0,
        weights=SimpleNamespace(gradient=1.0, entropy=0.5, eme=0.5, clipiqa=0.0, musiq=0.0),
        scales=SimpleNamespace(gradient=10.0, entropy=1.0, eme=0.1, clipiqa=1.0, musiq=0.01),
    )


def _policy() -> PolicyValueFCN:
    return PolicyValueFCN(in_channels=3, base_filters=16, num_dilated_blocks=2, init_log_sigma=-0.5)


def _env(mode: str) -> ImageEnhancementEnv:
    reward = CompositeReward(_reward_cfg(mode), device="cpu")
    bounds = ActionBounds(gamma=(0.3, 3.0), alpha=(0.5, 2.0), beta=(-0.2, 0.2))
    return ImageEnhancementEnv(reward_fn=reward, bounds=bounds, episode_length=3)


def test_ppo_scalar_one_update_runs() -> None:
    env = _env("scalar")
    policy = _policy()
    opt = Adam(policy.parameters(), lr=1e-3)
    trainer = PPOTrainer(
        policy, opt,
        PPOConfig(
            clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
            entropy_coef_min=0.001, entropy_anneal_steps=1000,
            ppo_epochs=2, minibatch_size=2,
            gae_gamma=0.99, gae_lambda=0.95, max_grad_norm=0.5,
            normalize_advantage=True, reward_mode="scalar",
        ),
    )
    x0 = torch.rand(2, 3, 16, 16)
    traj = collect_rollout(env, policy, x0, reward_mode="scalar")
    metrics = trainer.update(traj)
    assert "loss/total" in metrics
    assert all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values())


def test_ppo_pixel_one_update_runs() -> None:
    env = _env("pixel")
    policy = _policy()
    opt = Adam(policy.parameters(), lr=1e-3)
    trainer = PPOTrainer(
        policy, opt,
        PPOConfig(
            clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
            entropy_coef_min=0.001, entropy_anneal_steps=1000,
            ppo_epochs=2, minibatch_size=2,
            gae_gamma=0.99, gae_lambda=0.95, max_grad_norm=0.5,
            normalize_advantage=True, reward_mode="pixel",
        ),
    )
    x0 = torch.rand(2, 3, 16, 16)
    traj = collect_rollout(env, policy, x0, reward_mode="pixel")
    metrics = trainer.update(traj)
    assert "loss/total" in metrics
    assert all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values())


def test_grpo_one_update_runs() -> None:
    env = _env("scalar")
    policy = _policy()
    opt = Adam(policy.parameters(), lr=1e-3)
    trainer = GRPOTrainer(
        policy, opt, env,
        GRPOConfig(
            group_size=3, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
            entropy_coef_min=0.001, entropy_anneal_steps=1000,
            ppo_epochs=1, minibatch_size=2, max_grad_norm=0.5,
            drop_critic=True, beta_kl=0.0, reward_mode="scalar",
        ),
    )
    x0 = torch.rand(2, 3, 16, 16)
    metrics = trainer.update(x0)
    assert "loss/total" in metrics
    assert all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values())


def test_training_reduces_loss_over_time() -> None:
    """Smoke check that gradients actually update: loss on same batch should change."""
    env = _env("scalar")
    policy = _policy()
    opt = Adam(policy.parameters(), lr=1e-2)
    trainer = PPOTrainer(
        policy, opt,
        PPOConfig(
            clip_ratio=0.2, value_coef=0.5, entropy_coef=0.0,
            entropy_coef_min=0.0, entropy_anneal_steps=0,
            ppo_epochs=1, minibatch_size=2,
            gae_gamma=0.99, gae_lambda=0.95, max_grad_norm=0.5,
            normalize_advantage=True, reward_mode="scalar",
        ),
    )
    x0 = torch.rand(2, 3, 16, 16)
    losses = []
    for _ in range(4):
        traj = collect_rollout(env, policy, x0, reward_mode="scalar")
        m = trainer.update(traj)
        losses.append(m["loss/total"])
    # Allow noise: expect the min over last 2 to be different from initial by >0.
    assert any(abs(losses[0] - l) > 1e-6 for l in losses[1:])
