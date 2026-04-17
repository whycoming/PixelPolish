"""Save enhanced images + per-step action maps for qualitative inspection."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.env.image_env import ImageEnhancementEnv
from src.models.actions import (
    ActionBounds,
    apply_curve,
    raw_to_curve_params,
    sample_action,
)
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.composite import CompositeReward
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config, resolve_device


def _load_image(path: str, image_size: int, channels: int) -> torch.Tensor:
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    tx = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    x = tx(img)
    if channels == 3 and x.shape[0] == 1:
        x = x.expand(3, -1, -1).contiguous()
    if channels == 1 and x.shape[0] == 3:
        x = x.mean(dim=0, keepdim=True)
    return x.unsqueeze(0)  # [1, C, H, W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", default=None, action="append")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="path to a single image")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device or cfg.train.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    load_checkpoint(args.ckpt, policy, map_location=device)
    policy.eval()

    bounds = ActionBounds.from_config(cfg.action)
    reward_fn = CompositeReward(cfg.reward, device=device).to(device)
    env = ImageEnhancementEnv(reward_fn, bounds, episode_length=int(cfg.env.episode_length))

    x = _load_image(args.input, cfg.data.image_size, cfg.model.in_channels).to(device)
    save_image(x.clamp(0, 1), str(out_dir / "step_00_input.png"))

    x_t = env.reset(x)
    with torch.no_grad():
        for t in range(env.episode_length):
            mu, log_sigma, _ = policy(x_t)
            # Greedy: use mean (raw=mu) not samples, for deterministic visuals.
            gamma, alpha, beta = raw_to_curve_params(mu, bounds)
            x_t = apply_curve(x_t, gamma, alpha, beta)
            save_image(x_t.clamp(0, 1), str(out_dir / f"step_{t+1:02d}_image.png"))
            # Normalize each action channel to [0, 1] for visualization.
            def _norm(v: torch.Tensor) -> torch.Tensor:
                vmin, vmax = float(v.min()), float(v.max())
                if vmax - vmin < 1e-6:
                    return torch.zeros_like(v)
                return (v - vmin) / (vmax - vmin)
            save_image(_norm(gamma), str(out_dir / f"step_{t+1:02d}_gamma.png"))
            save_image(_norm(alpha), str(out_dir / f"step_{t+1:02d}_alpha.png"))
            save_image(_norm(beta), str(out_dir / f"step_{t+1:02d}_beta.png"))

    print(f"[visualize] wrote {env.episode_length * 4 + 1} files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
