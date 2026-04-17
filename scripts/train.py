"""End-to-end PPO / GRPO training entry point.

Typical usage:
    python -m scripts.train --config configs/base.yaml --data-root ./data

Override on 6 GB VRAM:
    python -m scripts.train --config configs/base.yaml --overrides configs/local_dev.yaml --data-root ./data
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.optim import Adam
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


def _grpo_config_from(cfg) -> GRPOConfig:
    t = cfg.train
    g = cfg.grpo
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
        reward_mode=str(cfg.reward.mode),
    )


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
    policy = PolicyValueFCN(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        num_dilated_blocks=cfg.model.num_dilated_blocks,
        init_log_sigma=cfg.model.init_log_sigma,
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=float(cfg.train.lr))

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
        grpo = GRPOTrainer(policy, optimizer, env, _grpo_config_from(cfg))

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
