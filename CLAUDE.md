# CLAUDE.md — RL-Based Universal Blind Image Enhancement

## Project Overview

Build a pixel-level reinforcement learning framework for **blind image enhancement without ground truth**, designed to generalize across image modalities (RGB, IR, grayscale). The agent learns to apply per-pixel tone-mapping curves guided by no-reference image quality rewards.

**Core idea**: treat each pixel as an RL agent sharing a single FCN policy; at each step, the agent predicts continuous curve parameters `[γ, α, β]` that transform the pixel; the policy is trained with PPO using a composite reward of NR-IQA models and physical image statistics — no paired GT needed.

**Starting point**: `FightingSrain/Pytorch-pixelRL` (clean PyTorch reimplementation of PixelRL). We replace its supervised reward and discrete filter actions with our NR reward and continuous curve actions.

---

## Design Decisions (Fixed)

These are settled. Do not revisit without explicit discussion.

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| **State** | Current image only, `[B, C, H, W]` | Simplicity. No degradation embedding, no step encoding. |
| **Action** | Continuous curve params `[γ, α, β]` per pixel | γ for nonlinear tone mapping, α·x+β for linear adjustment. Covers most tone-mapping needs. |
| **Reward** | Abstract interface; initial impl = `R_IQA + R_physics` | Open for later extension (task reward, PatchGAN, etc.). |
| **Agent granularity** | Pixel-level, FCN parameter sharing | Handles spatially non-uniform degradation. Standard PixelRL architecture. |
| **Episode** | Fixed K=5 steps, dense reward | No learned stopping. Reward given every step. |
| **Algorithm** | **PPO** (primary), GRPO (later for ablation) | PPO is the stable, well-documented choice. GRPO reserved for publication-stage comparison. |
| **Hardware target** | Single RTX 4090 (24GB) | Batch size and group size must fit. |

---

## Repository Layout

```
rl-enhance/
├── CLAUDE.md                     # This file
├── README.md                     # Usage instructions
├── requirements.txt
├── configs/
│   ├── base.yaml                 # Default config
│   └── ablation/                 # Experiment-specific overrides
├── src/
│   ├── models/
│   │   ├── policy_fcn.py         # Shared FCN backbone + policy/value heads
│   │   └── actions.py            # Curve application (γ, α, β)
│   ├── rewards/
│   │   ├── base.py               # RewardFunction abstract class
│   │   ├── iqa.py                # CLIP-IQA / MUSIQ wrappers (via pyiqa)
│   │   ├── physics.py            # Gradient, entropy, EME
│   │   └── composite.py          # Weighted combination
│   ├── env/
│   │   ├── image_env.py          # Gym-style env: apply action → new image + reward
│   │   └── degradation.py        # (Optional) synthetic degradation for pretraining
│   ├── algorithms/
│   │   ├── ppo.py                # PPO trainer
│   │   └── grpo.py               # (Phase 6) GRPO trainer
│   ├── data/
│   │   └── dataset.py            # Unpaired image dataset loader
│   └── utils/
│       ├── logging.py            # W&B / TensorBoard wrapper
│       └── checkpoints.py
├── scripts/
│   ├── train.py
│   ├── eval.py                   # Compute NIQE/BRISQUE/CLIP-IQA on test sets
│   └── visualize.py              # Save enhanced images + per-step action maps
└── tests/
    ├── test_actions.py           # Unit test: curves are monotonic, bounded
    ├── test_rewards.py           # Reward functions produce expected ranges
    └── test_training_loop.py     # One forward/backward pass on toy data
```

---

## Implementation Phases

Each phase is a checkpoint. **Do not start phase N+1 until phase N passes its validation criterion.** This prevents compounding bugs.

### Phase 1 — Skeleton (no RL yet)

**Goal**: get the data and model pipeline running end-to-end without training.

Tasks:
1. Clone `FightingSrain/Pytorch-pixelRL` as reference
2. Set up repo structure above
3. Implement `PolicyValueFCN`:
   - Dilated CNN backbone (dilations 1, 2, 3, 4 — same as PixelRL) for large receptive field
   - Policy head: output `[B, 3, H, W]` (μ for γ, α, β), passed through sigmoid, then range-mapped
   - Also output log σ per channel (learned standard deviation for Gaussian policy)
   - Value head: output `[B, 1, H, W]`
4. Implement `apply_curve`:
   - γ ∈ [0.3, 3.0], α ∈ [0.5, 2.0], β ∈ [-0.2, 0.2]
   - `x_new = clip(α · pow(x, γ) + β, 0, 1)`
   - Must be differentiable (even though not used for gradient flow in RL, we want it for debugging)
5. Dataset loader: unpaired images, any modality, normalize to [0, 1]

**Validation**:
- Run one forward pass on a batch of 4×256×256 images, get policy output
- Apply sampled action, verify output is also `[B, C, H, W]` in [0, 1]
- No NaN, no shape mismatches

### Phase 2 — Reward system

**Goal**: build the reward pipeline in isolation and verify it produces sensible signals.

Tasks:
1. Define `RewardFunction` abstract class:
   ```
   class RewardFunction(ABC):
       @abstractmethod
       def compute(self, x_prev: Tensor, x_curr: Tensor) -> Tensor:
           """Returns scalar or [B, 1, H, W] reward. Convention: higher is better."""
   ```
2. Install `pyiqa` (`pip install pyiqa`). Implement:
   - `CLIPIQAReward`: wraps `pyiqa.create_metric('clipiqa')`, frozen
   - `MUSIQReward`: wraps `pyiqa.create_metric('musiq')`, frozen (optional, keep CLIP-IQA as primary)
3. Implement `PhysicsReward`:
   - `avg_gradient`: mean of Sobel magnitude
   - `entropy`: normalized image entropy (histogram-based, differentiable approximation via soft binning)
   - `EME`: Agaian's measure of enhancement by entropy, block size 8
4. Implement `CompositeReward` that linearly combines sub-rewards with configurable weights
5. **Use relative reward**: `r_t = R(x_curr) - R(x_prev)`, not absolute. This is crucial for training stability.

**Validation**:
- Take a batch of test images
- Apply obviously-good enhancement (e.g., CLAHE) → reward should be positive
- Apply obviously-bad enhancement (e.g., γ=0.1 to crush blacks) → reward should be negative
- Compute reward on 100 natural images, check the distribution is reasonable (not saturated at one extreme)

**CRITICAL risk point**: CLIP-IQA is frozen; its gradient does NOT flow back to policy. Reward is used only as a scalar signal in PPO, so this is fine — but verify in code that `torch.no_grad()` wraps IQA model calls.

### Phase 3 — PPO trainer (scalar reward)

**Goal**: train the policy with PPO, reward treated as per-image scalar first (simplest).

Tasks:
1. Implement episode rollout:
   ```
   trajectory = []
   x_t = x_0
   for t in range(K=5):
       μ, log_σ = policy(x_t)        # [B, 3, H, W]
       a_t ~ Normal(μ, σ)            # reparameterize for sampling
       log_prob_t = log Normal.pdf(a_t | μ, σ).sum(pixel, channel)  # per-image
       x_t_next = apply_curve(x_t, a_t)
       r_t = reward(x_t, x_t_next)   # scalar per image
       v_t = value(x_t).mean()       # scalar per image
       trajectory.append((x_t, a_t, log_prob_t, r_t, v_t))
       x_t = x_t_next
   ```
2. Compute GAE advantage with γ=0.99, λ=0.95
3. PPO-clip loss with ε=0.2
4. Value loss: MSE to return
5. Entropy bonus: encourage exploration, coefficient 0.01 (anneal to 0 over training)
6. Multi-epoch update: K_epochs=4 per batch of collected trajectories
7. Normalize advantages per batch

**Validation**:
- Train on a small dataset (say 500 low-light images) for 10k steps
- Policy entropy should decrease over time (agent becoming more decisive)
- Reward should increase monotonically on a held-out validation set
- Visually inspect: enhanced images should look brighter/more contrasted, NOT blown out or crushed

### Phase 4 — Pixel-wise reward map

**Goal**: upgrade scalar reward to per-pixel reward map using PixelRL's reward_map_convolution technique.

Tasks:
1. Physics reward easily made spatial (gradient magnitude is already per-pixel, entropy via sliding window)
2. IQA reward is inherently global — broadcast scalar to full map as fallback, or use a proxy like local CLIP similarity
3. Implement `reward_map_convolution`:
   - For each agent (pixel), consider neighborhood reward: `r̃_i = conv(r, weight)` with fixed weights that decay with distance
4. Update PPO's advantage computation to operate per-pixel:
   - GAE is computed per pixel independently
   - Log probs are summed over the action dimensions (3 for γ, α, β) but kept per-pixel
   - Loss averaged over pixels and batch

**Validation**:
- Compare training stability vs Phase 3 scalar reward
- Visualize per-step action maps (color-coded by action magnitude) — should show spatially-varying decisions, not uniform
- Final enhancement quality (NIQE, CLIP-IQA on held-out set) should improve over Phase 3

### Phase 5 — Multi-modality evaluation

**Goal**: verify generalization claim. The whole point of this design is modality-agnostic enhancement.

Tasks:
1. Train on a mixed dataset: RGB low-light (LOL) + grayscale (or converted) + IR (FLIR or similar)
2. Evaluate on each modality separately with:
   - NIQE, BRISQUE (NR-IQA, standard)
   - CLIP-IQA (what we trained on — should be highest, sanity check)
   - Visual inspection
3. Ablations:
   - Train RGB-only, test on IR → measure performance drop (expected to be noticeable)
   - Train mixed, test on each → should be the strongest result
4. Compare against baselines:
   - Zero-DCE / Zero-DCE++ (supervised-ish competitor)
   - ReLLIE (closest RL-based method)
   - Histogram equalization (classical baseline)

**Validation**:
- Mixed-trained model is within 10% of modality-specific trained model on each modality
- No catastrophic failures on any modality (no pure-black outputs, no color shifts on RGB)

### Phase 6 — (Optional) GRPO ablation

**Goal**: add GRPO as an alternative trainer, compare head-to-head with PPO.

Tasks:
1. Implement `grpo.py` in `algorithms/`. Reuse policy/value/reward code. Only the trainer differs.
2. Core mechanics:
   - Group size G=8 (tune within 4-16 by memory)
   - For each input image, roll out G independent trajectories
   - Compute returns per trajectory
   - Group z-normalize: `A_g = (R_g - mean_g) / (std_g + eps)`
   - PPO-clip loss using group advantage (critic optional; GRPO can drop it)
3. Memory: B×G×K forward passes per update. With B=2, G=8, K=5, H=W=256, this is 80 forwards per step. Use gradient accumulation if needed.

**Validation**:
- Compare PPO vs GRPO on identical reward and data
- GRPO's training curve should be smoother in high-noise reward settings (e.g., when IQA weight dominates)
- If GRPO is NOT clearly better, report honestly — this is a useful negative result

---

## Coding Standards

- **Type hints** on all public functions and class methods
- **Docstrings** (one-liner minimum) on every function, explaining shape of inputs/outputs
- **Shape assertions** at boundaries (tensor passes between modules), e.g., `assert x.ndim == 4`
- **Config-driven**: no magic numbers in training code. All hyperparameters in YAML.
- **Logging**: use W&B if available, TensorBoard fallback. Log at minimum:
  - Total reward (mean over batch)
  - Each sub-reward individually (critical for diagnosing reward hacking)
  - Policy entropy
  - Value loss, policy loss
  - Sample enhanced images every N=500 steps
- **Unit tests** for non-trivial logic (action application, reward computation, advantage calculation). Run with `pytest` before merging.
- **Deterministic seeding** in all tests

---

## Key Risks and Mitigations

Listed in order of likelihood × impact.

### Risk 1: Reward hacking (HIGH likelihood)
Agent learns to exploit CLIP-IQA by producing grayish/over-smoothed images that score high but look bad.

**Mitigation**:
- Always use relative reward `Δ R` not absolute
- Log all sub-rewards separately; if CLIP-IQA saturates while physics drops, that's the signal
- Add a `R_penalty` term early (color shift, over-smoothing) if hacking emerges
- Visual inspection every 500 steps is non-negotiable

### Risk 2: Policy entropy collapse (MEDIUM)
Policy becomes deterministic too fast, stops exploring.

**Mitigation**:
- Entropy bonus with coefficient 0.01, don't anneal to zero — floor at 0.001
- Monitor log σ — if σ → 0 everywhere, policy is collapsing

### Risk 3: Pixel-wise value network is hard to train (MEDIUM)
Value estimates per pixel are high-variance.

**Mitigation**:
- Use GAE with λ=0.95 (reduces variance vs λ=1)
- Warmup: freeze value network for first 1k steps, use MC returns as baseline
- If all else fails, drop critic entirely (go advantage-free, just use normalized returns)

### Risk 4: Action bounds are too restrictive (LOW)
γ ∈ [0.3, 3.0] may not cover extreme cases.

**Mitigation**:
- Keep bounds as config parameters, easy to widen later
- Monitor what fraction of sampled actions hit the boundary — if >10%, bounds are too tight

### Risk 5: Training set modality imbalance (MEDIUM in Phase 5)
If 90% RGB and 10% IR, policy will be RGB-centric.

**Mitigation**:
- Balanced sampling: equal probability per modality per batch, regardless of dataset size
- Modality-specific learning rates if needed (but try without first)

---

## Out of Scope

To keep this project focused. If these come up, resist scope creep:

- **Supervised pretraining**: this is a pure RL-from-scratch project
- **Diffusion-based priors**: different paradigm entirely
- **Per-modality specialized tricks** (e.g., IR column-median filter): violates universality goal
- **GAN-based discriminator rewards**: was considered, dropped for simplification
- **Video enhancement**: images only

---

## References (for reading, not copying)

- PixelRL (Furuta et al., TMM 2020) — architecture blueprint
- ReLLIE (Zhang et al., ACM MM 2021) — reward design inspiration
- ALL-E+ (TPAMI 2025) — aesthetic reward concept
- Zero-DCE (Guo et al., CVPR 2020) — non-reference loss functions we adapt
- PPO (Schulman et al., 2017) — algorithm
- GRPO (DeepSeek, 2024) — for Phase 6

---

## Success Criteria (Project Level)

The project is considered successful if:

1. **Functional**: PPO trainer runs stably on single RTX 4090 for >24 hours without crashes
2. **Quantitative**: Beats Zero-DCE on NIQE by >5% on cross-modality test (train mixed, test IR)
3. **Qualitative**: Enhanced images are visually improved on 80%+ of held-out samples (blind user study of 50 images)
4. **Reproducible**: Full training run reproducible from config + seed, <1% variance in final metric across 3 seeds

Failure modes acceptable at project end:
- GRPO turns out no better than PPO → honest negative finding
- Cross-modality transfer is imperfect but not catastrophic → expected given state simplification

Not acceptable:
- Policy collapses to identity or constant output
- Any sub-reward becomes dominant and causes visible artifacts
- Training requires >72h on 4090 for one run (indicates pipeline inefficiency)
