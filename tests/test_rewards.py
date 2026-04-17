import math
from types import SimpleNamespace

import torch

from src.rewards.composite import CompositeReward
from src.rewards.physics import EMEReward, EntropyReward, GradientReward


def _reward_cfg(mode: str = "scalar", relative: bool = True, pyiqa: bool = False):
    return SimpleNamespace(
        mode=mode,
        relative=relative,
        pixel_smooth_radius=1,
        weights=SimpleNamespace(
            gradient=1.0,
            entropy=0.5,
            eme=0.5,
            clipiqa=1.0 if pyiqa else 0.0,
            musiq=0.0,
        ),
        scales=SimpleNamespace(
            gradient=10.0, entropy=1.0, eme=0.1, clipiqa=1.0, musiq=0.01
        ),
    )


def test_gradient_reward_per_pixel_shape() -> None:
    g = GradientReward()
    x = torch.rand(2, 3, 16, 16)
    r = g._compute(x)
    assert r.shape == (2, 1, 16, 16)
    assert torch.all(r >= 0.0)


def test_entropy_between_0_and_1() -> None:
    e = EntropyReward()
    x = torch.rand(3, 3, 32, 32)
    r = e._compute(x)
    assert r.shape == (3,)
    assert torch.all(r >= 0.0) and torch.all(r <= 1.0 + 1e-4)


def test_entropy_higher_for_more_varied_image() -> None:
    e = EntropyReward()
    uniform = torch.full((1, 1, 32, 32), 0.5)
    varied = torch.rand(1, 1, 32, 32)
    assert float(e._compute(varied)) > float(e._compute(uniform))


def test_eme_higher_for_contrasty_image() -> None:
    eme = EMEReward(block=8)
    # Constant image -> ratio ~1 -> eme ~0
    flat = torch.full((1, 1, 32, 32), 0.5)
    # Checkerboard -> high contrast
    checker = torch.zeros(1, 1, 32, 32)
    checker[..., ::2, ::2] = 1.0
    checker[..., 1::2, 1::2] = 1.0
    assert float(eme._compute(checker)) > float(eme._compute(flat))


def test_composite_scalar_shape_and_sign_on_known_good_enhancement() -> None:
    cfg = _reward_cfg(mode="scalar", relative=True)
    reward = CompositeReward(cfg, device="cpu")

    # Dark, low-contrast image (simulated low-light).
    x_prev = torch.rand(2, 3, 32, 32) * 0.3
    # A simple "good" enhancement: brighten and increase contrast.
    x_curr = (x_prev * 2.0 + 0.1).clamp(0.0, 1.0)

    r, per_name = reward.compute(x_prev, x_curr)
    assert r.shape == (2,)
    # Most sub-rewards should prefer the enhanced image.
    assert float(r.mean()) > 0.0
    assert set(per_name.keys()) == {"gradient", "entropy", "eme"}


def test_composite_scalar_negative_on_obvious_bad_enhancement() -> None:
    cfg = _reward_cfg(mode="scalar", relative=True)
    reward = CompositeReward(cfg, device="cpu")
    x_prev = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    # Crush everything to near-black: bad.
    x_curr = (x_prev * 0.05).clamp(0.0, 1.0)
    r, _ = reward.compute(x_prev, x_curr)
    assert float(r.mean()) < 0.0


def test_composite_pixel_mode_returns_pixel_map() -> None:
    cfg = _reward_cfg(mode="pixel", relative=True)
    reward = CompositeReward(cfg, device="cpu")
    x_prev = torch.rand(2, 3, 16, 16)
    x_curr = torch.rand(2, 3, 16, 16)
    r, _ = reward.compute(x_prev, x_curr)
    assert r.shape == (2, 1, 16, 16)


def test_composite_raises_if_all_weights_zero() -> None:
    cfg = SimpleNamespace(
        mode="scalar",
        relative=True,
        pixel_smooth_radius=0,
        weights=SimpleNamespace(gradient=0.0, entropy=0.0, eme=0.0, clipiqa=0.0, musiq=0.0),
        scales=SimpleNamespace(gradient=1.0, entropy=1.0, eme=1.0, clipiqa=1.0, musiq=1.0),
    )
    try:
        CompositeReward(cfg, device="cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError when no rewards are enabled")
