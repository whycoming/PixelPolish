# PixelPolish

**Pixel-level reinforcement learning for blind image enhancement — no ground truth, any modality.**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-research-orange)

PixelPolish trains a fully-convolutional policy that predicts a per-pixel tone-mapping curve `y = clip(α · x^γ + β, 0, 1)` and iteratively "polishes" an image over K=5 steps. The policy is optimized with **PPO** against a composite **no-reference** reward (CLIP-IQA / MUSIQ / gradient / entropy / EME) — no paired clean targets required. Because the state is just the image and the action is a generic tone curve, the *same* policy works on RGB, grayscale, and IR inputs.

<p align="center"><i>Low-light RGB → Enhanced. IR → Enhanced. Same policy, no fine-tune.</i></p>

---

## Why pixel-level RL?

| Classical regression | Diffusion / GAN | PixelPolish |
| --- | --- | --- |
| Needs paired (degraded, clean) data | Needs large priors / distilled teachers | Needs *only* unpaired images |
| One model per degradation type | One model per modality | One model per *everything* |
| Deterministic output | Stochastic, often hallucinatory | Curve-based → minimally invasive, reversible |

The policy is shared across pixels (FCN), so the receptive field of dilated convs lets each pixel condition its local action on neighborhood context — uniform or spatially varying, up to you.

## Features

- **Modality-agnostic**: train on a mix of RGB / grayscale / IR, evaluate anywhere.
- **No ground truth**: composite NR-IQA reward (CLIP-IQA, MUSIQ) + physics-based (Sobel gradient, soft-histogram entropy, Agaian EME).
- **Two reward modes**: scalar per-image (Phase 3) or per-pixel reward map with spatial GAE (Phase 4).
- **PPO by default, GRPO for ablation**: swap via one config line.
- **Config-driven**: `configs/base.yaml` is the single source of truth; `configs/local_dev.yaml` lets the same code run on a 6 GB laptop GPU.
- **Type-hinted, shape-asserted, pytest-covered**: 30+ unit / integration tests, CPU-runnable.
- **Soft pyiqa dependency**: the physics-only subset of rewards needs no external models; CLIP-IQA / MUSIQ / NIQE / BRISQUE auto-enable if `pyiqa` is installed.

## Architecture at a glance

```
           ┌──────────────────────────── PolicyValueFCN ────────────────────────────┐
 x_t  ─▶   │ Conv3 → ReLU → [Dilated3 d=1,2,3,4 × ReLU]₄ → Conv3 → ReLU            │
 [B,C,H,W] │                  │                                    │                 │
           │                  ├──── 1×1 → μ        [B,3,H,W]       │                 │
           │                  ├──── 1×1 → log σ    [B,3,H,W]       │                 │
           │                  └──── 1×1 → V        [B,1,H,W]       │                 │
           └────────────────────────────────────────────────────────┘
                                         │ sample
                                         ▼
                          tanh + affine → (γ, α, β)
                                         │ apply_curve
                                         ▼
                                     x_{t+1}
                                         │ reward = R(x_{t+1}) − R(x_t)
                                         ▼
                              PPO / GRPO update
```

Core design decisions (more detail in `CLAUDE.md`):

| Dimension | Choice |
| --- | --- |
| State | Current image only `[B, C, H, W]` |
| Action | Continuous per-pixel `[γ ∈ [0.3,3], α ∈ [0.5,2], β ∈ [-0.2,0.2]]` |
| Reward | Composite NR, **relative** `ΔR` per step |
| Episode | Fixed K=5, dense reward |
| Agent | Pixel-level, FCN parameter sharing (PixelRL-style) |
| Algorithm | PPO (primary) / GRPO (optional ablation) |

## Install

```bash
git clone https://github.com/<you>/pixelpolish.git
cd pixelpolish
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / remote
source .venv/bin/activate

pip install -r requirements.txt
```

Requires PyTorch ≥ 2.1 and, optionally, `pyiqa` for CLIP-IQA / MUSIQ / NIQE / BRISQUE.

## Quick start

### 1. Smoke test (CPU or GPU, no data needed)

```bash
python -m scripts.smoke_forward --config configs/base.yaml --overrides configs/local_dev.yaml
```

Expected: tensor-shape dump + `SMOKE TEST OK`. This is the Phase-1 validation gate from `CLAUDE.md`.

### 2. Train

**Remote GPU (RTX 4090, 24 GB):**
```bash
python -m scripts.train \
    --config configs/base.yaml \
    --data-root /path/to/images \
    --output-dir runs --run-name exp01
```

**Laptop GPU (6 GB VRAM):**
```bash
python -m scripts.train \
    --config configs/base.yaml \
    --overrides configs/local_dev.yaml \
    --data-root ./data/small \
    --output-dir runs --run-name local01
```

**Per-pixel reward map (Phase 4):**
```bash
python -m scripts.train \
    --config configs/base.yaml \
    --overrides configs/ablation/ppo_pixel.yaml \
    --data-root /path/to/images
```

**GRPO ablation (Phase 6):**
```bash
python -m scripts.train \
    --config configs/base.yaml \
    --overrides configs/ablation/grpo.yaml \
    --data-root /path/to/images
```

CLI dotlist overrides are supported too:
```bash
--set train.lr=1e-4 --set train.batch_size=4
```

### 3. Evaluate

```bash
python -m scripts.eval \
    --config configs/base.yaml \
    --ckpt runs/exp01/checkpoints/final.pt \
    --data-root /path/to/test
```

Reports NIQE, BRISQUE, CLIP-IQA, MUSIQ (via `pyiqa`) plus built-in gradient / entropy / EME, with per-metric `(input, output, Δ)`.

### 4. Visualize

```bash
python -m scripts.visualize \
    --config configs/base.yaml \
    --ckpt runs/exp01/checkpoints/final.pt \
    --input path/to/image.png \
    --output outputs/vis01
```

Writes the input, each intermediate step `x_{t+1}`, and normalized γ / α / β action maps per step.

### 5. Run tests

```bash
pytest -q
```

All 30+ tests run on CPU in a few seconds — `pyiqa` is **not** required.

## Project layout

```
pixelpolish/
├── CLAUDE.md                 # Design doc (start here)
├── configs/
│   ├── base.yaml             # Full hyperparameters
│   ├── local_dev.yaml        # 6 GB VRAM override
│   └── ablation/
│       ├── ppo_pixel.yaml    # Phase 4: per-pixel reward map
│       └── grpo.yaml         # Phase 6: GRPO
├── src/
│   ├── models/
│   │   ├── policy_fcn.py     # Dilated-CNN backbone + μ/logσ/V heads
│   │   └── actions.py        # Curve bounds, sampling, log-prob, entropy, apply_curve
│   ├── rewards/
│   │   ├── base.py           # RewardFunction ABC + RelativeReward
│   │   ├── physics.py        # Gradient / Entropy / EME
│   │   ├── iqa.py            # pyiqa wrappers (CLIP-IQA, MUSIQ, NIQE, BRISQUE)
│   │   └── composite.py      # Weighted composition, scalar / pixel mode
│   ├── env/
│   │   ├── image_env.py      # Gym-style batched env
│   │   └── degradation.py    # Optional synthetic degradations
│   ├── algorithms/
│   │   ├── rollout.py        # Episode collection with bootstrap value
│   │   ├── advantage.py      # GAE (shape-generic)
│   │   ├── ppo.py            # PPOTrainer, scalar + pixel
│   │   └── grpo.py           # GRPOTrainer, group z-norm
│   ├── data/
│   │   └── dataset.py        # UnpairedImageDataset + MixedModalityDataset
│   └── utils/
│       ├── config.py         # OmegaConf loader with overrides
│       ├── logging.py        # TensorBoard wrapper
│       ├── checkpoints.py    # save / load / prune
│       └── seed.py
├── scripts/
│   ├── smoke_forward.py      # Phase-1 gate
│   ├── train.py              # Main entry
│   ├── eval.py               # NR-IQA metric sweep
│   └── visualize.py          # Save action maps + enhanced steps
└── tests/
    ├── test_actions.py
    ├── test_policy_fcn.py
    ├── test_rewards.py
    ├── test_dataset.py
    ├── test_smoke_forward.py
    └── test_training_loop.py
```

## Configuration cheat sheet

Key knobs in `configs/base.yaml`:

```yaml
action:         # per-pixel curve bounds
  gamma_range: [0.3, 3.0]
  alpha_range: [0.5, 2.0]
  beta_range:  [-0.2, 0.2]

reward:
  mode: scalar                # or 'pixel' for Phase 4
  relative: true              # ΔR per step (recommended)
  weights: {gradient: 1.0, entropy: 0.5, eme: 0.5, clipiqa: 1.0, musiq: 0.0}
  scales:  {gradient: 10.0, entropy: 1.0, eme: 0.1, clipiqa: 1.0, musiq: 0.01}

train:
  algorithm: ppo              # or 'grpo'
  batch_size: 8
  episode_length: 5
  ppo_epochs: 4
  clip_ratio: 0.2
  entropy_coef: 0.01          # annealed to entropy_coef_min
  gae_gamma: 0.99
  gae_lambda: 0.95
```

## Roadmap (from `CLAUDE.md`)

- [x] **Phase 1** — Skeleton: FCN policy + curve actions + smoke pass
- [x] **Phase 2** — Composite reward (physics + CLIP-IQA / MUSIQ), relative deltas
- [x] **Phase 3** — PPO with scalar reward, GAE, entropy bonus
- [x] **Phase 4** — Per-pixel reward map + spatial GAE
- [ ] **Phase 5** — Multi-modality benchmark (RGB low-light + grayscale + IR) vs Zero-DCE / ReLLIE / HE
- [x] **Phase 6** — GRPO ablation (group z-norm, optional drop critic)

## Known limitations & risks

- **Reward hacking.** CLIP-IQA can be exploited by over-smoothing. Mitigations: relative rewards, log all sub-rewards separately, qualitative inspection every N updates. If CLIP-IQA saturates while physics drops → stop and investigate.
- **Entropy collapse.** `log σ` is floored via config annealing (`entropy_coef_min`); watch it on TensorBoard.
- **Pixel-wise value variance.** GAE λ=0.95 helps. If unstable, try `grpo` with `drop_critic=true` — no critic, group-normalized returns only.

## Citation

If this helps your research, please cite:

```bibtex
@software{pixelpolish2026,
  title  = {PixelPolish: Pixel-Level RL for Blind Image Enhancement},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<you>/pixelpolish}
}
```

## References

- Furuta *et al.*, **PixelRL**: Fully Convolutional Network with Reinforcement Learning for Image Processing, IEEE TMM 2020
- Schulman *et al.*, **PPO**: Proximal Policy Optimization, arXiv:1707.06347
- DeepSeek, **GRPO**: Group Relative Policy Optimization, 2024
- Wang *et al.*, **CLIP-IQA**: Exploring CLIP for Assessing the Look and Feel of Images, AAAI 2023
- Ke *et al.*, **MUSIQ**: Multi-Scale Image Quality Transformer, ICCV 2021
- Zhang *et al.*, **ReLLIE**: Deep Reinforcement Learning for Customized Low-Light Image Enhancement, ACM MM 2021
- Guo *et al.*, **Zero-DCE**: Zero-Reference Deep Curve Estimation, CVPR 2020

## License

MIT. See `LICENSE`.
