"""Evaluate a trained policy on a test directory, reporting NR-IQA metrics.

Uses pyiqa for NIQE/BRISQUE/CLIP-IQA. If pyiqa is not installed we only
report our in-repo physics metrics (gradient, entropy, EME).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import UnpairedImageDataset
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import ActionBounds, apply_curve, sample_action
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.composite import CompositeReward
from src.rewards.physics import EMEReward, EntropyReward, GradientReward
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config, resolve_device


def _physics_metrics(x: torch.Tensor) -> Dict[str, float]:
    grad = GradientReward()._compute(x).mean()
    ent = EntropyReward()._compute(x).mean()
    eme = EMEReward()._compute(x).mean()
    return {
        "gradient": float(grad),
        "entropy": float(ent),
        "eme": float(eme),
    }


def _iqa_metrics(x: torch.Tensor, device: str) -> Dict[str, float]:
    try:
        import pyiqa  # type: ignore
    except ImportError:
        return {}
    results: Dict[str, float] = {}
    for name in ("niqe", "brisque", "clipiqa", "musiq"):
        try:
            m = pyiqa.create_metric(name, device=device, as_loss=False)
            with torch.no_grad():
                v = m(x.to(device))
            if torch.is_tensor(v):
                v = float(v.mean())
            results[name] = float(v)
        except Exception as exc:  # pragma: no cover
            print(f"[eval] skip {name}: {exc}")
    return results


def _enhance_batch(policy, env: ImageEnhancementEnv, x0: torch.Tensor, bounds: ActionBounds) -> torch.Tensor:
    """Run one deterministic-ish episode (sample actions greedily via mean)."""
    x_t = env.reset(x0)
    with torch.no_grad():
        for _ in range(env.episode_length):
            mu, log_sigma, _ = policy(x_t)
            # Greedy action = mean; use sample_action for shape consistency but pass near-zero sigma.
            gamma, alpha, beta, _, _ = sample_action(mu, log_sigma.clamp_max(-5.0), bounds)
            x_t = apply_curve(x_t, gamma, alpha, beta)
    return x_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", default=None, action="append")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device or cfg.train.device)
    image_size = int(args.image_size or cfg.data.image_size)

    policy = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    load_checkpoint(args.ckpt, policy, map_location=device)
    policy.eval()

    # Dummy composite reward, not used for metrics — only to instantiate env.
    reward_fn = CompositeReward(cfg.reward, device=device).to(device)
    env = ImageEnhancementEnv(
        reward_fn=reward_fn,
        bounds=ActionBounds.from_config(cfg.action),
        episode_length=int(cfg.env.episode_length),
    )

    dataset = UnpairedImageDataset(
        root=args.data_root, image_size=image_size, channels=cfg.model.in_channels
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sum_in: Dict[str, float] = {}
    sum_out: Dict[str, float] = {}
    count = 0

    for batch in tqdm(loader, desc="eval"):
        x = batch.to(device)
        y = _enhance_batch(policy, env, x, ActionBounds.from_config(cfg.action))

        m_in = {**_physics_metrics(x), **_iqa_metrics(x, device)}
        m_out = {**_physics_metrics(y), **_iqa_metrics(y, device)}
        b = x.shape[0]
        for k, v in m_in.items():
            sum_in[k] = sum_in.get(k, 0.0) + v * b
        for k, v in m_out.items():
            sum_out[k] = sum_out.get(k, 0.0) + v * b
        count += b

    print(f"\n=== Evaluation over {count} images ===")
    print(f"{'metric':<12} {'input':>12} {'output':>12} {'delta':>12}")
    keys = sorted(set(sum_in) | set(sum_out))
    for k in keys:
        a = sum_in.get(k, 0.0) / max(1, count)
        b = sum_out.get(k, 0.0) / max(1, count)
        print(f"{k:<12} {a:>12.4f} {b:>12.4f} {b - a:>+12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
