"""End-to-end PPO / GRPO training entry point.

Typical usage:
    python -m scripts.train --config configs/base.yaml --data-root ./data

Override on 6 GB VRAM:
    python -m scripts.train --config configs/base.yaml --overrides configs/local_dev.yaml --data-root ./data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.algorithms.grpo import GRPOConfig, GRPOTrainer
from src.algorithms.ppo import PPOConfig, PPOTrainer
from src.algorithms.rollout import collect_rollout
from src.data.dataset import MixedModalityDataset, UnpairedImageDataset
from src.env.image_env import ImageEnhancementEnv
from src.models.actions import ActionBounds
from src.models.policy_fcn import PolicyValueFCN
from src.rewards.composite import CompositeReward
from src.utils.checkpoints import prune_old_checkpoints, save_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.logging import Logger
from src.utils.seed import seed_everything


def _build_dataset(cfg, data_root: Optional[str]):
    if cfg.data.modalities:
        mods: List[Tuple[float, str]] = [
            (float(w), str(p)) for (w, p) in cfg.data.modalities
        ]
        return MixedModalityDataset(
            modalities=mods,
            image_size=cfg.data.image_size,
            channels=cfg.data.channels,
        )
    root = data_root if data_root is not None else cfg.data.root
    return UnpairedImageDataset(
        root=root,
        image_size=cfg.data.image_size,
        channels=cfg.data.channels,
    )


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _ppo_config_from(cfg) -> PPOConfig:
    t = cfg.train
    return PPOConfig(
        clip_ratio=float(t.clip_ratio),
        value_coef=float(t.value_coef),
        entropy_coef=float(t.entropy_coef),
        entropy_coef_min=float(t.entropy_coef_min),
        entropy_anneal_steps=int(t.entropy_anneal_steps),
        ppo_epochs=int(t.ppo_epochs),
        minibatch_size=int(t.minibatch_size),
        gae_gamma=float(t.gae_gamma),
        gae_lambda=float(t.gae_lambda),
        max_grad_norm=float(t.max_grad_norm),
        normalize_advantage=bool(t.normalize_advantage),
        reward_mode=str(cfg.reward.mode),
    )


def _load_iqa_targets(path: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Load (μ, σ) per head from a precompute_iqa_target.py JSON, if provided."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"[train] warn: iqa_target_path={path} not found, falling back to Δ-Borda")
        return None
    with open(p, "r") as f:
        blob = json.load(f)
    out: Dict[str, Tuple[float, float]] = {}
    for name, stats in blob.get("heads", {}).items():
        mu = float(stats["mu"])
        sigma = float(stats["sigma"])
        if sigma < 1e-6:
            print(f"[train] warn: head '{name}' σ~0 in target file, skipping")
            continue
        out[name] = (mu, sigma)
    if not out:
        print(f"[train] warn: iqa_target file '{path}' contained no usable heads")
        return None
    print(f"[train] loaded IQA targets from {path}: "
          + ", ".join(f"{k} μ={v[0]:.3f} σ={v[1]:.3f}" for k, v in out.items()))
    return out


def _grpo_config_from(cfg) -> GRPOConfig:
    t = cfg.train
    g = cfg.grpo
    reward_mode = str(cfg.reward.mode)
    borda_heads: List[str] = []
    iqa_targets: Optional[Dict[str, Tuple[float, float]]] = None
    state_aug_iqa: List[str] = []
    if reward_mode == "terminal_borda":
        borda_heads = [str(h) for h in (getattr(cfg.reward, "borda_heads", None) or [])]
        iqa_targets = _load_iqa_targets(getattr(cfg.reward, "iqa_target_path", None))
        state_aug_iqa = [str(h) for h in (getattr(g, "state_aug_iqa", None) or [])]
    return GRPOConfig(
        group_size=int(g.group_size),
        clip_ratio=float(t.clip_ratio),
        value_coef=float(t.value_coef),
        entropy_coef=float(t.entropy_coef),
        entropy_coef_min=float(t.entropy_coef_min),
        entropy_anneal_steps=int(t.entropy_anneal_steps),
        ppo_epochs=int(t.ppo_epochs),
        minibatch_size=int(t.minibatch_size),
        max_grad_norm=float(t.max_grad_norm),
        drop_critic=bool(g.drop_critic),
        beta_kl=float(g.beta_kl),
        init_log_sigma_ref=float(getattr(g, "init_log_sigma_ref", cfg.model.init_log_sigma)),
        reward_mode=reward_mode,
        borda_heads=borda_heads,
        shared_global_noise=bool(getattr(g, "shared_global_noise", False)),
        iqa_targets=iqa_targets,
        state_aug_iqa=state_aug_iqa,
    )


def _build_iqa_heads_for_borda(cfg, device: str):
    """Instantiate the IQA heads named in `cfg.reward.borda_heads`. Returns
    a dict {name: callable(x_0, x_K)->[B]} with unified binary signature.
    Unary heads (clipiqa, musiq, l_exposure, gray_world) ignore x_0."""
    from src.rewards.iqa import build_head, as_binary_score
    names = [str(h) for h in (getattr(cfg.reward, "borda_heads", None) or [])]
    extra: Dict[str, dict] = {
        "l_exposure": {
            "patch_size": int(getattr(cfg.reward, "exposure_patch_size", 16)),
            "target": float(getattr(cfg.reward, "exposure_target", 0.6)),
        },
        "lpips": {"net": str(getattr(cfg.reward, "lpips_net", "vgg"))},
    }
    heads = {}
    for name in names:
        h = build_head(name, device=device, **extra.get(name, {}))
        if h is None:
            print(f"[train] warn: borda head '{name}' unavailable, skipping.")
            continue
        heads[name] = as_binary_score(h)
    return heads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--overrides", default=None, action="append",
                    help="repeatable path to override YAML")
    ap.add_argument("--set", default=None, action="append",
                    help="repeatable CLI override in `a.b=c` form")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides, cli_overrides=args.set)
    if args.run_name is not None:
        cfg.log.run_name = args.run_name
    if args.output_dir is not None:
        cfg.log.dir = args.output_dir

    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.train.device)

    # --- Dataset / loader ---
    dataset = _build_dataset(cfg, args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        drop_last=True,
        pin_memory=(device != "cpu"),
    )
    data_iter = _infinite_loader(loader)

    # --- Model / optimizer ---
    # Auto-size in_channels when GRPO state augmentation is on: each entry in
    # grpo.state_aug_iqa adds one broadcast channel to the policy input.
    base_in = int(cfg.model.in_channels)
    algorithm_name = str(cfg.train.algorithm).lower()
    extra_in = 0
    if algorithm_name == "grpo" and str(cfg.reward.mode) == "terminal_borda":
        extra_in = len([h for h in (getattr(cfg.grpo, "state_aug_iqa", None) or [])])
    policy = PolicyValueFCN(
        in_channels=base_in + extra_in,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    if extra_in:
        print(f"[train] policy in_channels = {base_in} + {extra_in} "
              f"(state_aug_iqa={list(cfg.grpo.state_aug_iqa)})")
    optimizer = AdamW(
        policy.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(getattr(cfg.train, "weight_decay", 0.01)),
    )

    # --- Reward / env ---
    reward_fn = CompositeReward(cfg.reward, device=device).to(device)
    bounds = ActionBounds.from_config(cfg.action)
    env = ImageEnhancementEnv(
        reward_fn=reward_fn,
        bounds=bounds,
        episode_length=int(cfg.env.episode_length),
    )

    # --- Logger ---
    run_dir = Path(cfg.log.dir) / str(cfg.log.run_name)
    logger = Logger(str(run_dir), backend=str(cfg.log.backend))
    ckpt_dir = Path(cfg.checkpoint.dir) if cfg.checkpoint.dir else run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional resume ---
    start_update = 0
    if args.resume is not None:
        from src.utils.checkpoints import load_checkpoint
        obj = load_checkpoint(args.resume, policy, optimizer, map_location=device)
        start_update = int(obj.get("step", 0))
        print(f"[resume] loaded {args.resume} at update={start_update}")

    algorithm = str(cfg.train.algorithm).lower()
    assert algorithm in {"ppo", "grpo"}, f"unknown algorithm: {algorithm}"

    if algorithm == "ppo":
        ppo = PPOTrainer(policy, optimizer, _ppo_config_from(cfg))
    else:
        grpo_cfg = _grpo_config_from(cfg)
        iqa_heads = (
            _build_iqa_heads_for_borda(cfg, device)
            if grpo_cfg.reward_mode == "terminal_borda"
            else None
        )
        grpo = GRPOTrainer(policy, optimizer, env, grpo_cfg, iqa_heads=iqa_heads)

    total_updates = int(cfg.train.total_updates)
    log_interval = int(cfg.log.interval)
    img_interval = int(cfg.log.image_interval)
    num_log_images = int(cfg.log.num_log_images)
    ckpt_interval = int(cfg.checkpoint.interval)

    pbar = tqdm(range(start_update, total_updates), desc="train", dynamic_ncols=True)
    for update in pbar:
        x0 = next(data_iter).to(device, non_blocking=True)

        if algorithm == "ppo":
            traj = collect_rollout(env, policy, x0, reward_mode=cfg.reward.mode)
            policy.train()
            metrics = ppo.update(traj)
            # Attach per-step mean rewards for logging.
            if cfg.reward.mode == "scalar":
                metrics["reward/episode_mean"] = float(traj.rewards.sum(dim=0).mean())
            else:
                metrics["reward/episode_mean"] = float(
                    traj.rewards.sum(dim=0).mean()
                )
            for k, vals in traj.sub_rewards.items():
                metrics[f"reward/sub/{k}"] = float(sum(vals) / max(1, len(vals)))
        else:
            policy.train()
            metrics = grpo.update(x0)

        if update % log_interval == 0:
            logger.log_scalars(metrics, step=update)
            pbar.set_postfix({
                "total": f"{metrics['loss/total']:.3f}",
                "ent": f"{metrics['loss/entropy']:.3f}",
                "kl": f"{metrics['policy/kl_approx']:.4f}",
            })

        if img_interval > 0 and update % img_interval == 0 and algorithm == "ppo":
            with torch.no_grad():
                sample_x0 = x0[:num_log_images]
                traj_vis = collect_rollout(env, policy, sample_x0, reward_mode=cfg.reward.mode)
                final = traj_vis.states[-1]  # before last action
                # One more step to see enhancement applied.
                mu, log_sigma, _ = policy(final)
                from src.models.actions import raw_to_curve_params, apply_curve, sample_action
                gamma, alpha, beta, _, _ = sample_action(mu, log_sigma, bounds)
                final_out = apply_curve(final, gamma, alpha, beta)
                logger.log_images("samples/input", sample_x0, step=update)
                logger.log_images("samples/output", final_out, step=update)

        if ckpt_interval > 0 and update > 0 and update % ckpt_interval == 0:
            path = str(ckpt_dir / f"update_{update:07d}.pt")
            save_checkpoint(path, policy, optimizer, step=update)
            prune_old_checkpoints(str(ckpt_dir), keep_last=int(cfg.checkpoint.keep_last))

    final_path = str(ckpt_dir / "final.pt")
    save_checkpoint(final_path, policy, optimizer, step=total_updates)
    logger.close()
    print(f"[train] done. final checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
