"""Diagnostic for v4 (target-distribution Borda) policy.

For each test image: (1) load image at 256, (2) run K=5 greedy enhancement,
(3) report mean brightness and clipiqa/musiq for input vs output.

Goal: verify the policy
  - keeps already-good images near identity (low Δ brightness, IQA stays near μ)
  - still brightens dark images toward target (Δ brightness positive, IQA → μ)

NOT a replacement for v3 comparison (v3 ckpt was lost), but enough to judge whether
target-distribution reward fixed the over-brightening / over-darkening pattern.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from src.models.actions import ActionBounds, apply_curve, raw_to_curve_params
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.iqa import build_head
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config, resolve_device


def _load_image(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, nargs="+",
                    help="paths to images to evaluate (can be globs expanded by shell)")
    ap.add_argument("--out-dir", default=None,
                    help="if set, save input+output PNGs here")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)

    policy = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    load_checkpoint(args.ckpt, policy, map_location=device)
    policy.eval()
    bounds = ActionBounds.from_config(cfg.action)

    heads = {}
    for n in ["clipiqa", "musiq"]:
        h = build_head(n, device=device)
        if h is not None:
            heads[n] = h._compute

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    K = int(cfg.env.episode_length)
    print(f"\n{'image':25s}  {'b_in':>6s} -> {'b_out':>6s}  "
          f"{'clipiqa_in':>10s} -> {'clipiqa_out':>11s}  "
          f"{'musiq_in':>9s} -> {'musiq_out':>10s}")
    print("-" * 100)

    for p_str in args.images:
        p = Path(p_str)
        if not p.is_file():
            continue
        x0 = _load_image(p, cfg.data.image_size).to(device)
        b_in = float(x0.mean()) * 255.0
        with torch.no_grad():
            iqa_in = {n: float(fn(x0)) for n, fn in heads.items()}
            x = x0
            for _ in range(K):
                mu, log_sigma, _ = policy(x)
                # Greedy: use μ (raw=μ) — no exploration noise.
                gamma, alpha, beta = raw_to_curve_params(mu, bounds)
                x = apply_curve(x, gamma, alpha, beta)
            b_out = float(x.mean()) * 255.0
            iqa_out = {n: float(fn(x)) for n, fn in heads.items()}

        print(f"{p.name[:25]:25s}  "
              f"{b_in:6.1f} -> {b_out:6.1f}  "
              f"{iqa_in.get('clipiqa', float('nan')):10.4f} -> {iqa_out.get('clipiqa', float('nan')):11.4f}  "
              f"{iqa_in.get('musiq', float('nan')):9.2f} -> {iqa_out.get('musiq', float('nan')):10.2f}")

        if out_dir is not None:
            from torchvision.utils import save_image
            save_image(x0.clamp(0, 1), str(out_dir / f"{p.stem}_in.png"))
            save_image(x.clamp(0, 1), str(out_dir / f"{p.stem}_v4.png"))

    print("-" * 100)
    print(f"target μ: clipiqa=0.764  musiq=63.49  (BSDS500 reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
