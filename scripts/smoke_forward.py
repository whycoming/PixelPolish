"""Phase-1 smoke test: forward + action + curve application, no training.

Runs on the configured device (falls back to CPU when CUDA unavailable).
"""

import argparse
import sys

import torch

from src.models.actions import ActionBounds, apply_curve, sample_action
from src.models.policy_fcn import PolicyValueFCN
from src.utils.config import load_config, resolve_device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--device", default=None, help="override cfg.train.device")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device or cfg.train.device)

    net = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)

    bounds = ActionBounds.from_config(cfg.action)
    x = torch.rand(
        cfg.train.batch_size, cfg.model.in_channels,
        cfg.data.image_size, cfg.data.image_size,
        device=device,
    )
    with torch.no_grad():
        mu, log_sigma, value = net(x)
        gamma, alpha, beta, log_prob, raw = sample_action(mu, log_sigma, bounds)
        x_next = apply_curve(x, gamma, alpha, beta)

    def _rng(t):  # pragma: no cover
        return f"[{float(t.min()):.3f}, {float(t.max()):.3f}]"

    print(f"device:     {device}")
    print(f"x:          {tuple(x.shape)} {_rng(x)}")
    print(f"mu:         {tuple(mu.shape)} {_rng(mu)}")
    print(f"log_sigma:  {tuple(log_sigma.shape)} {_rng(log_sigma)}")
    print(f"value:      {tuple(value.shape)} {_rng(value)}")
    print(f"gamma:      {tuple(gamma.shape)} {_rng(gamma)}")
    print(f"alpha:      {tuple(alpha.shape)} {_rng(alpha)}")
    print(f"beta:       {tuple(beta.shape)} {_rng(beta)}")
    print(f"log_prob:   {tuple(log_prob.shape)} mean={float(log_prob.mean()):.3f}")
    print(f"x_next:     {tuple(x_next.shape)} {_rng(x_next)}")

    assert torch.isfinite(x_next).all(), "x_next has NaN/Inf"
    assert x_next.min() >= 0.0 and x_next.max() <= 1.0, "x_next out of [0,1]"
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
