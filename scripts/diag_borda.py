"""Diagnostic: replay G=8 GRPO rollouts from a checkpoint and report
per-head Δ-scores + rank agreement across {clipiqa, musiq, niqe}.

If the 3 heads disagree (different best sibling, low Spearman), Borda's
average rank degenerates → near-zero advantage → no learning signal.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.algorithms.rollout import collect_rollout_actions_only
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import ActionBounds
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.composite import CompositeReward
from src.rewards.iqa import build_head
from src.utils.config import load_config


def _load_image(path: str, size: int) -> Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float((ra * rb).sum() / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shared-noise", action="store_true",
                    help="use shared_global_noise sampling (one [B,3,1,1] eps per rollout)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device

    policy = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    obj = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = obj.get("model_state", obj.get("state_dict", obj))
    policy.load_state_dict(sd, strict=False)
    policy.eval()

    bounds = ActionBounds.from_config(cfg.action)
    reward_fn = CompositeReward(cfg.reward, device=device).to(device)
    env = ImageEnhancementEnv(reward_fn=reward_fn, bounds=bounds,
                              episode_length=int(cfg.env.episode_length))

    head_names = list(cfg.reward.borda_heads)
    heads = {}
    for n in head_names:
        h = build_head(n, device=device)
        if h is not None:
            heads[n] = h._compute
    print(f"[diag] heads loaded: {list(heads.keys())}")

    x0 = _load_image(args.input, int(cfg.data.image_size)).to(device)

    # Score baseline once.
    s0 = {n: float(fn(x0).view(-1)[0]) for n, fn in heads.items()}
    print(f"[diag] baseline scores (higher=better, sign-flipped for niqe):")
    for n, v in s0.items():
        print(f"    {n:10s}: {v:+.4f}")

    # G rollouts.
    G = int(args.group_size)
    finals = []
    for g in range(G):
        traj = collect_rollout_actions_only(
            env, policy, x0, shared_global_noise=bool(args.shared_noise)
        )
        finals.append(traj.final_state)
    print(f"[diag] sampling mode: {'shared_global_noise' if args.shared_noise else 'per_pixel_noise'}")

    # Score each.
    deltas = {n: np.zeros(G, dtype=np.float64) for n in heads}
    abs_scores = {n: np.zeros(G, dtype=np.float64) for n in heads}
    for g, fs in enumerate(finals):
        for n, fn in heads.items():
            v = float(fn(fs).view(-1)[0])
            abs_scores[n][g] = v
            deltas[n][g] = v - s0[n]

    print(f"\n[diag] per-rollout Δ (delta from baseline) over G={G}:")
    print(f"    {'g':>2} | " + " | ".join(f"{n:>10s}" for n in heads))
    for g in range(G):
        row = " | ".join(f"{deltas[n][g]:+10.4f}" for n in heads)
        print(f"    {g:2d} | {row}")

    print(f"\n[diag] per-head Δ statistics:")
    print(f"    {'head':10s}  {'mean':>9s}  {'std':>9s}  {'min':>9s}  {'max':>9s}  {'std/|mean|':>10s}")
    for n in heads:
        d = deltas[n]
        ratio = d.std() / (abs(d.mean()) + 1e-9)
        print(f"    {n:10s}  {d.mean():+9.4f}  {d.std():9.4f}  {d.min():+9.4f}  {d.max():+9.4f}  {ratio:10.2f}")

    # Borda ranks (higher Δ → higher rank G-1..0).
    print(f"\n[diag] per-head ranks (G-1=best):")
    ranks = {}
    for n in heads:
        order = (-deltas[n]).argsort()  # descending
        r = np.empty(G, dtype=np.int64)
        for i, g in enumerate(order):
            r[g] = G - 1 - i
        ranks[n] = r
    print(f"    {'g':>2} | " + " | ".join(f"{n:>10s}" for n in heads) + " | avg_rank | adv")
    avg_rank = np.mean([ranks[n] for n in heads], axis=0)
    adv = avg_rank - avg_rank.mean()
    for g in range(G):
        row = " | ".join(f"{ranks[n][g]:10d}" for n in heads)
        print(f"    {g:2d} | {row} | {avg_rank[g]:8.2f} | {adv[g]:+5.2f}")

    print(f"\n[diag] Borda advantage range: [{adv.min():+.2f}, {adv.max():+.2f}]  "
          f"std={adv.std():.3f} (pure noise expected ~1.5; 0 = total cancellation)")

    print(f"\n[diag] pairwise Spearman rank correlation between heads:")
    names = list(heads)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = _spearman(deltas[names[i]], deltas[names[j]])
            verdict = "AGREE" if rho > 0.5 else ("OPPOSE" if rho < -0.3 else "MIXED")
            print(f"    {names[i]:>10s} vs {names[j]:<10s}: rho={rho:+.3f}  [{verdict}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
