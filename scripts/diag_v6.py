"""Diagnostic for v6 (input-anchored Borda + state-augmented policy).

For each image:
  1) Load at cfg.data.image_size, build state-aug channels = IQA(x_0) broadcast
  2) Greedy K-step rollout using raw=μ (no noise)
  3) Report brightness and each head's score for x_0 and x_K
  4) Optionally save input + output PNGs

Key differences from diag_v4: policy takes 3 + |state_aug_iqa| channels; all
Borda heads are called with unified (x_0, x_K) signature.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from src.models.actions import ActionBounds, apply_curve, raw_to_curve_params
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.iqa import as_binary_score, build_head
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config, resolve_device


def _load_image(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _build_aug(heads: Dict, names, x0: torch.Tensor) -> torch.Tensor:
    """IQA(x_0) broadcast channels. musiq normalised by /100."""
    chans = []
    with torch.no_grad():
        for n in names:
            s = heads[n](x0, x0).view(-1)
            if n == "musiq":
                s = s / 100.0
            b, _, h, w = x0.shape
            chans.append(s.view(b, 1, 1, 1).expand(b, 1, h, w))
    return torch.cat(chans, dim=1).to(x0.device).to(x0.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", default=None, action="append")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device)

    # Build all heads we need: scoring + state-aug.
    head_names = list(cfg.reward.borda_heads or [])
    aug_names = list(getattr(cfg.grpo, "state_aug_iqa", None) or [])
    needed = sorted(set(head_names) | set(aug_names))
    heads: Dict[str, object] = {}
    extra = {"lpips": {"net": str(getattr(cfg.reward, "lpips_net", "vgg"))}}
    for n in needed:
        h = build_head(n, device=device, **extra.get(n, {}))
        if h is None:
            print(f"[diag] warn: head '{n}' unavailable, skipping.")
            continue
        heads[n] = as_binary_score(h)

    # Policy: in_channels auto-sized 3 + len(aug_names).
    policy = PolicyValueFCN(
        in_channels=int(cfg.model.in_channels) + len(aug_names),
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    load_checkpoint(args.ckpt, policy, map_location=device)
    policy.eval()
    bounds = ActionBounds.from_config(cfg.action)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    K = int(cfg.env.episode_length)
    width = 10
    hdr = f"{'image':25s}  {'b_in':>6s}->{'b_out':>6s}  " + "  ".join(
        f"{n[:width]:>{width}s}[in→out]" for n in head_names
    )
    print(hdr)
    print("-" * len(hdr))

    for p_str in args.images:
        p = Path(p_str)
        if not p.is_file():
            continue
        x0 = _load_image(p, cfg.data.image_size).to(device)
        aug = _build_aug(heads, aug_names, x0) if aug_names else None

        with torch.no_grad():
            scores_in = {n: float(heads[n](x0, x0)) for n in head_names if n in heads}
            x = x0
            for _ in range(K):
                pin = x if aug is None else torch.cat([x, aug], dim=1)
                mu, log_sigma, _ = policy(pin)
                gamma, alpha, beta = raw_to_curve_params(mu, bounds)
                x = apply_curve(x, gamma, alpha, beta)
            scores_out = {n: float(heads[n](x0, x)) for n in head_names if n in heads}

        b_in = float(x0.mean()) * 255.0
        b_out = float(x.mean()) * 255.0
        row = f"{p.name[:25]:25s}  {b_in:6.1f}->{b_out:6.1f}  "
        for n in head_names:
            si = scores_in.get(n, float("nan"))
            so = scores_out.get(n, float("nan"))
            row += f"{si:+.3f}→{so:+.3f}  "
        print(row)

        if out_dir is not None:
            from torchvision.utils import save_image
            save_image(x0.clamp(0, 1), str(out_dir / f"{p.stem}_in.png"))
            save_image(x.clamp(0, 1), str(out_dir / f"{p.stem}_v6.png"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
