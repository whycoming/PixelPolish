import math

import torch

from src.models.actions import (
    ActionBounds,
    apply_curve,
    evaluate_log_prob,
    gaussian_entropy,
    raw_to_curve_params,
    sample_action,
)


def _bounds() -> ActionBounds:
    return ActionBounds(gamma=(0.3, 3.0), alpha=(0.5, 2.0), beta=(-0.2, 0.2))


def test_apply_curve_identity_params() -> None:
    x = torch.rand(1, 3, 16, 16)
    gamma = torch.ones(1, 1, 16, 16)
    alpha = torch.ones(1, 1, 16, 16)
    beta = torch.zeros(1, 1, 16, 16)
    y = apply_curve(x, gamma, alpha, beta)
    assert torch.allclose(y, x.clamp(0.0, 1.0), atol=1e-6)


def test_apply_curve_output_in_unit_range() -> None:
    x = torch.rand(2, 3, 16, 16)
    gamma = torch.rand(2, 1, 16, 16) * 2.7 + 0.3
    alpha = torch.rand(2, 1, 16, 16) * 1.5 + 0.5
    beta = (torch.rand(2, 1, 16, 16) * 0.4) - 0.2
    y = apply_curve(x, gamma, alpha, beta)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
    assert y.shape == x.shape


def test_apply_curve_monotonic_in_x() -> None:
    # For α>0 and γ>0, the unclamped curve is non-decreasing in x. Check before clipping extremes.
    x1 = torch.linspace(0.1, 0.7, steps=100).reshape(1, 1, 10, 10).expand(1, 3, 10, 10).contiguous()
    x2 = x1 + 1e-3
    gamma = torch.full_like(x1[:, :1], 1.5)
    alpha = torch.full_like(x1[:, :1], 1.2)
    beta = torch.full_like(x1[:, :1], 0.05)
    y1 = apply_curve(x1, gamma, alpha, beta)
    y2 = apply_curve(x2, gamma, alpha, beta)
    assert torch.all(y2 + 1e-6 >= y1)


def test_raw_to_curve_params_at_extremes() -> None:
    bounds = _bounds()
    raw = torch.tensor([[-10.0, -10.0, -10.0]]).reshape(1, 3, 1, 1)
    gamma, alpha, beta = raw_to_curve_params(raw, bounds)
    assert math.isclose(float(gamma), bounds.gamma[0], abs_tol=1e-3)
    assert math.isclose(float(alpha), bounds.alpha[0], abs_tol=1e-3)
    assert math.isclose(float(beta), bounds.beta[0], abs_tol=1e-3)
    raw = torch.tensor([[10.0, 10.0, 10.0]]).reshape(1, 3, 1, 1)
    gamma, alpha, beta = raw_to_curve_params(raw, bounds)
    assert math.isclose(float(gamma), bounds.gamma[1], abs_tol=1e-3)
    assert math.isclose(float(alpha), bounds.alpha[1], abs_tol=1e-3)
    assert math.isclose(float(beta), bounds.beta[1], abs_tol=1e-3)


def test_sample_action_shapes_and_bounds() -> None:
    bounds = _bounds()
    mu = torch.zeros(2, 3, 4, 4)
    log_sigma = torch.full_like(mu, -0.5)
    gamma, alpha, beta, log_prob, raw = sample_action(mu, log_sigma, bounds)
    for t, (lo, hi) in zip((gamma, alpha, beta), (bounds.gamma, bounds.alpha, bounds.beta)):
        assert t.shape == (2, 1, 4, 4)
        assert torch.all(t >= lo - 1e-5)
        assert torch.all(t <= hi + 1e-5)
    assert log_prob.shape == (2, 1, 4, 4)
    assert torch.isfinite(log_prob).all()
    assert raw.shape == (2, 3, 4, 4)


def test_evaluate_log_prob_matches_sample_at_sample_time() -> None:
    mu = torch.randn(1, 3, 3, 3)
    log_sigma = torch.randn(1, 3, 3, 3).clamp(-2.0, 0.5)
    _, _, _, lp_at_sample, raw = sample_action(mu, log_sigma, _bounds())
    lp_reeval = evaluate_log_prob(raw, mu, log_sigma)
    assert torch.allclose(lp_at_sample, lp_reeval, atol=1e-6)


def test_gaussian_entropy_matches_closed_form() -> None:
    log_sigma = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]]])  # sigma = 1
    ent = gaussian_entropy(log_sigma)
    # H = 3 * 0.5 * log(2 pi e) for unit-variance Gaussian in 3D
    expected = 3.0 * 0.5 * math.log(2.0 * math.pi * math.e)
    assert math.isclose(float(ent.item()), expected, rel_tol=1e-5)
