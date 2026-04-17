import torch

from src.models.policy_fcn import PolicyValueFCN


def _net() -> PolicyValueFCN:
    return PolicyValueFCN(in_channels=3, base_filters=16, num_dilated_blocks=4, init_log_sigma=-0.5)


def test_forward_shapes() -> None:
    net = _net()
    x = torch.rand(2, 3, 16, 16)
    mu, log_sigma, value = net(x)
    assert mu.shape == (2, 3, 16, 16)
    assert log_sigma.shape == (2, 3, 16, 16)
    assert value.shape == (2, 1, 16, 16)


def test_forward_finite_and_log_sigma_bounded() -> None:
    net = _net()
    x = torch.rand(1, 3, 32, 32)
    mu, log_sigma, value = net(x)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_sigma).all()
    assert torch.isfinite(value).all()
    assert log_sigma.min() >= -5.0 - 1e-3
    assert log_sigma.max() <= 2.0 + 1e-3


def test_initial_log_sigma_matches_bias() -> None:
    net = PolicyValueFCN(in_channels=3, base_filters=16, num_dilated_blocks=2, init_log_sigma=-0.3)
    x = torch.rand(1, 3, 8, 8)
    _, log_sigma, _ = net(x)
    # Heads are zero-inited so output equals bias everywhere.
    assert torch.allclose(log_sigma, torch.full_like(log_sigma, -0.3), atol=1e-5)


def test_accepts_various_spatial_sizes() -> None:
    net = _net()
    for size in (16, 24, 33, 64):
        x = torch.rand(1, 3, size, size)
        mu, _, value = net(x)
        assert mu.shape == (1, 3, size, size)
        assert value.shape == (1, 1, size, size)
