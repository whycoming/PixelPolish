"""Tests for terminal-Borda GRPO additions: log-uniform actions, Borda
rank advantage, KL-to-identity, and rollout-without-rewards path."""

import math

import torch

from src.algorithms.grpo import (
    GRPOConfig,
    GRPOTrainer,
    _borda_rank_advantage,
    _kl_to_identity,
)
from src.algorithms.rollout import collect_rollout_actions_only
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import ActionBounds, raw_to_curve_params
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.physics import GradientReward
from src.rewards.base import RewardFunction


class _ConstReward(RewardFunction):
    spatial = False

    def _compute(self, x):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def test_log_uniform_raw_zero_maps_to_identity() -> None:
    bounds = ActionBounds(
        gamma=(0.5, 2.0),
        alpha=(2 ** -0.5, 2 ** 0.5),
        beta=(-0.1, 0.1),
        gamma_log=True,
        alpha_log=True,
    )
    raw = torch.zeros(1, 3, 4, 4)
    g, a, b = raw_to_curve_params(raw, bounds)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6), f"γ at raw=0 = {g.mean()}"
    assert torch.allclose(a, torch.ones_like(a), atol=1e-6), f"α at raw=0 = {a.mean()}"
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-6), f"β at raw=0 = {b.mean()}"


def test_log_uniform_extremes_match_bounds() -> None:
    bounds = ActionBounds(gamma=(0.5, 2.0), alpha=(0.5, 2.0), beta=(-0.1, 0.1),
                          gamma_log=True, alpha_log=True)
    raw = torch.tensor([[10.0, 10.0, 10.0]]).reshape(1, 3, 1, 1)
    g, a, b = raw_to_curve_params(raw, bounds)
    assert math.isclose(float(g), 2.0, rel_tol=1e-3)
    assert math.isclose(float(a), 2.0, rel_tol=1e-3)
    raw = torch.tensor([[-10.0, -10.0, -10.0]]).reshape(1, 3, 1, 1)
    g, a, b = raw_to_curve_params(raw, bounds)
    assert math.isclose(float(g), 0.5, rel_tol=1e-3)
    assert math.isclose(float(a), 0.5, rel_tol=1e-3)


def test_borda_rank_advantage_centered_and_correct() -> None:
    # G=4 rollouts, B=2 batch, 2 heads.
    # Head A scores: rollout 0 best, 3 worst.
    # Head B scores: same order. Result should be deterministic ranks.
    head_scores = {
        "A": torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
        "B": torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
    }
    adv = _borda_rank_advantage(head_scores)
    # Rank 3 -> best, 0 -> worst. Centered means -1.5, -0.5, 0.5, 1.5 reversed: 1.5, 0.5, -0.5, -1.5
    expected = torch.tensor([[1.5, 1.5], [0.5, 0.5], [-0.5, -0.5], [-1.5, -1.5]])
    assert torch.allclose(adv, expected, atol=1e-5), f"got {adv}"
    # Centered: column sums are zero.
    assert torch.allclose(adv.sum(dim=0), torch.zeros(2), atol=1e-5)


def test_borda_robust_to_single_head_outlier() -> None:
    # Head A says rollout 0 wins; Head B and C disagree (rollout 0 worst).
    # Borda majority -> rollout 0 should NOT win.
    G, B = 4, 1
    sA = torch.tensor([[10.0], [0.0], [0.0], [0.0]])              # A: 0 best
    sB = torch.tensor([[0.0], [3.0], [2.0], [1.0]])               # B: 1 best
    sC = torch.tensor([[0.0], [3.0], [2.0], [1.0]])               # C: 1 best
    adv = _borda_rank_advantage({"A": sA, "B": sB, "C": sC})
    # Rollout 1 should have highest advantage (best on 2 of 3 heads).
    assert adv[1, 0] > adv[0, 0], f"adv 1 ({adv[1,0]}) should beat adv 0 ({adv[0,0]})"


def test_kl_to_identity_zero_at_reference() -> None:
    # μ=0, log_σ=log_σ_ref ⇒ KL = 0
    mu = torch.zeros(2, 3, 4, 4)
    log_sigma_ref = -0.5
    log_sigma = torch.full_like(mu, log_sigma_ref)
    kl = _kl_to_identity(mu, log_sigma, log_sigma_ref)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_to_identity_positive_when_off_reference() -> None:
    mu = torch.full((1, 3, 2, 2), 1.0)
    log_sigma = torch.zeros(1, 3, 2, 2)  # σ=1
    kl = _kl_to_identity(mu, log_sigma, log_sigma_ref=-0.5)
    assert torch.all(kl > 0), f"KL should be > 0 when off reference, got {kl}"


def test_rollout_actions_only_shapes_and_final_state() -> None:
    policy = PolicyValueFCN(in_channels=3, base_filters=8, num_dilated_blocks=1)
    bounds = ActionBounds(gamma=(0.5, 2.0), alpha=(0.5, 2.0), beta=(-0.1, 0.1),
                          gamma_log=True, alpha_log=True)
    env = ImageEnhancementEnv(reward_fn=_ConstReward(), bounds=bounds, episode_length=3)
    x0 = torch.rand(2, 3, 16, 16)
    traj = collect_rollout_actions_only(env, policy, x0)
    assert traj.states.shape == (3, 2, 3, 16, 16)
    assert traj.raw_actions.shape == (3, 2, 3, 16, 16)
    assert traj.log_probs.shape == (3, 2, 1, 16, 16)
    assert traj.final_state is not None and traj.final_state.shape == x0.shape
    assert torch.all(traj.final_state >= 0.0) and torch.all(traj.final_state <= 1.0)


def test_grpo_terminal_borda_smoke() -> None:
    """One full update with a fake IQA head ensures wiring is correct."""
    torch.manual_seed(0)
    policy = PolicyValueFCN(in_channels=3, base_filters=8, num_dilated_blocks=1,
                            init_log_sigma=-0.5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bounds = ActionBounds(gamma=(0.5, 2.0), alpha=(0.5, 2.0), beta=(-0.1, 0.1),
                          gamma_log=True, alpha_log=True)
    env = ImageEnhancementEnv(reward_fn=_ConstReward(), bounds=bounds, episode_length=3)
    cfg = GRPOConfig(
        group_size=4,
        ppo_epochs=1,
        minibatch_size=2,
        drop_critic=True,
        beta_kl=0.05,
        init_log_sigma_ref=-0.5,
        reward_mode="terminal_borda",
        borda_heads=["fake_a", "fake_b"],
    )
    fake_grad = GradientReward()
    iqa_heads = {
        "fake_a": lambda x: fake_grad._compute(x).mean(dim=(1, 2, 3)),
        "fake_b": lambda x: x.mean(dim=(1, 2, 3)),
    }
    trainer = GRPOTrainer(policy, optimizer, env, cfg, iqa_heads=iqa_heads)
    x0 = torch.rand(2, 3, 16, 16)
    metrics = trainer.update(x0)
    assert "loss/total" in metrics
    assert "loss/kl_ref" in metrics
    assert metrics["loss/kl_ref"] >= 0.0
    assert "borda/score_fake_a" in metrics
    assert "borda/score_fake_b" in metrics
    # In Δ-mode (no iqa_targets), per-head xK diagnostics should also be logged.
    assert "borda/xk_fake_a_mean" in metrics
    assert "borda/xk_fake_b_mean" in metrics


def test_grpo_terminal_borda_target_mode_smoke() -> None:
    """Target-distribution mode: scores = -((s_K - μ)/σ)²; Borda still works."""
    torch.manual_seed(0)
    policy = PolicyValueFCN(in_channels=3, base_filters=8, num_dilated_blocks=1,
                            init_log_sigma=-0.5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bounds = ActionBounds(gamma=(0.5, 2.0), alpha=(0.5, 2.0), beta=(-0.1, 0.1),
                          gamma_log=True, alpha_log=True)
    env = ImageEnhancementEnv(reward_fn=_ConstReward(), bounds=bounds, episode_length=3)
    cfg = GRPOConfig(
        group_size=4,
        ppo_epochs=1,
        minibatch_size=2,
        drop_critic=True,
        beta_kl=0.05,
        init_log_sigma_ref=-0.5,
        reward_mode="terminal_borda",
        borda_heads=["fake_a", "fake_b"],
        iqa_targets={"fake_a": (0.5, 0.1), "fake_b": (0.4, 0.05)},
    )
    iqa_heads = {
        "fake_a": lambda x: x.mean(dim=(1, 2, 3)),
        "fake_b": lambda x: x.std(dim=(1, 2, 3)),
    }
    trainer = GRPOTrainer(policy, optimizer, env, cfg, iqa_heads=iqa_heads)
    x0 = torch.rand(2, 3, 16, 16)
    metrics = trainer.update(x0)
    # Scores are -(z²), so always <= 0.
    assert metrics["borda/score_fake_a"] <= 0.0
    assert metrics["borda/score_fake_b"] <= 0.0


def test_local_exposure_score_at_target_is_max() -> None:
    """Constant-luma image at E=0.6 must score 0; off-target must be negative."""
    from src.rewards.exposure import LocalExposureReward
    head = LocalExposureReward(patch_size=4, target=0.6)
    x_at = torch.full((2, 3, 16, 16), 0.6)
    x_off = torch.full((2, 3, 16, 16), 0.1)
    s_at = head._compute(x_at)
    s_off = head._compute(x_off)
    assert s_at.shape == (2,)
    assert torch.allclose(s_at, torch.zeros(2), atol=1e-5), f"at-target score {s_at}"
    assert torch.all(s_off < s_at), f"off-target {s_off} should be < at-target {s_at}"
    # Symmetry: 0.1 below and 0.1 above target should score equally.
    x_above = torch.full((2, 3, 16, 16), 0.7)
    x_below = torch.full((2, 3, 16, 16), 0.5)
    assert torch.allclose(head._compute(x_above), head._compute(x_below), atol=1e-5)


def test_grpo_terminal_borda_mixed_target_and_delta_modes() -> None:
    """3 heads: clipiqa/musiq fakes use target-mode, l_exposure uses Δ-mode."""
    torch.manual_seed(0)
    from src.rewards.exposure import LocalExposureReward
    policy = PolicyValueFCN(in_channels=3, base_filters=8, num_dilated_blocks=1,
                            init_log_sigma=-0.5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bounds = ActionBounds(gamma=(0.5, 2.0), alpha=(0.5, 2.0), beta=(-0.1, 0.1),
                          gamma_log=True, alpha_log=True)
    env = ImageEnhancementEnv(reward_fn=_ConstReward(), bounds=bounds, episode_length=3)
    cfg = GRPOConfig(
        group_size=4,
        ppo_epochs=1,
        minibatch_size=2,
        drop_critic=True,
        beta_kl=0.05,
        init_log_sigma_ref=-0.5,
        reward_mode="terminal_borda",
        borda_heads=["fake_a", "fake_b", "l_exposure"],
        # Only the two fake heads have targets; l_exposure falls through to Δ-mode.
        iqa_targets={"fake_a": (0.5, 0.1), "fake_b": (0.4, 0.05)},
    )
    exp_head = LocalExposureReward(patch_size=4, target=0.6)
    iqa_heads = {
        "fake_a": lambda x: x.mean(dim=(1, 2, 3)),
        "fake_b": lambda x: x.std(dim=(1, 2, 3)),
        "l_exposure": exp_head._compute,
    }
    trainer = GRPOTrainer(policy, optimizer, env, cfg, iqa_heads=iqa_heads)
    x0 = torch.rand(2, 3, 16, 16)
    metrics = trainer.update(x0)
    assert metrics["borda/score_fake_a"] <= 0.0      # target-mode (always ≤0)
    assert metrics["borda/score_fake_b"] <= 0.0      # target-mode (always ≤0)
    assert "borda/score_l_exposure" in metrics       # Δ-mode (sign unconstrained)
    assert "borda/xk_l_exposure_mean" in metrics
    # KL still tracked when 3 heads are active.
    assert metrics["loss/kl_ref"] >= 0.0
