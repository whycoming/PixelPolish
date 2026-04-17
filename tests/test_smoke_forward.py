import torch

from src.models.actions import ActionBounds, apply_curve, sample_action
from src.models.policy_fcn import PolicyValueFCN


def test_end_to_end_forward_valid() -> None:
    bounds = ActionBounds(gamma=(0.3, 3.0), alpha=(0.5, 2.0), beta=(-0.2, 0.2))
    net = PolicyValueFCN(in_channels=3, base_filters=16, num_dilated_blocks=4, init_log_sigma=-0.5)
    net.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        mu, log_sigma, value = net(x)
        gamma, alpha, beta, log_prob, _ = sample_action(mu, log_sigma, bounds)
        x_next = apply_curve(x, gamma, alpha, beta)
    assert x_next.shape == x.shape
    assert torch.isfinite(x_next).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(log_prob).all()
    assert x_next.min() >= 0.0 and x_next.max() <= 1.0


def test_k_step_episode_stays_valid() -> None:
    bounds = ActionBounds(gamma=(0.3, 3.0), alpha=(0.5, 2.0), beta=(-0.2, 0.2))
    net = PolicyValueFCN(in_channels=3, base_filters=16, num_dilated_blocks=4, init_log_sigma=-0.5)
    net.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        for _ in range(5):
            mu, log_sigma, _ = net(x)
            gamma, alpha, beta, _, _ = sample_action(mu, log_sigma, bounds)
            x = apply_curve(x, gamma, alpha, beta)
            assert torch.isfinite(x).all()
            assert x.min() >= 0.0 and x.max() <= 1.0
