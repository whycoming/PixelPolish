"""Weighted composition of sub-rewards with scale-then-weight pipeline.

Config shape (see `configs/base.yaml`):

    reward:
      mode: scalar | pixel
      pixel_smooth_radius: 2
      relative: true
      weights: { gradient, entropy, eme, clipiqa, musiq }
      scales:  { gradient, entropy, eme, clipiqa, musiq }

Final reward = sum_k weight_k * scale_k * (R_k(x_curr) - R_k(x_prev))    [if relative]
or            sum_k weight_k * scale_k *  R_k(x_curr)                    [otherwise]

Shape returned:
  - mode=scalar  -> [B]
  - mode=pixel   -> [B, 1, H, W]
Spatially-atomic rewards (entropy, EME, IQA) are broadcast across H, W in pixel
mode; gradient is inherently spatial. An optional Gaussian smoothing
(`pixel_smooth_radius`) stabilizes the per-pixel reward map.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.rewards.base import RelativeReward, RewardFunction
from src.rewards.physics import EMEReward, EntropyReward, GradientReward


@dataclass
class _SubReward:
    name: str
    fn: RewardFunction
    weight: float
    scale: float


def _load_iqa(name: str, device: str) -> Optional[RewardFunction]:
    """Lazily import pyiqa-backed rewards. Returns None if pyiqa missing."""
    try:
        from src.rewards.iqa import (  # noqa: WPS433
            BRISQUEReward,
            CLIPIQAReward,
            MUSIQReward,
            NIQEReward,
        )
    except Exception:
        return None
    try:
        if name == "clipiqa":
            return CLIPIQAReward(device=device)
        if name == "musiq":
            return MUSIQReward(device=device)
        if name == "niqe":
            return NIQEReward(device=device)
        if name == "brisque":
            return BRISQUEReward(device=device)
    except ImportError:
        return None
    return None


def _get(node, name: str, default: float = 0.0) -> float:
    """Read an attribute from an OmegaConf DictConfig or a plain namespace/dict."""
    if node is None:
        return float(default)
    if hasattr(node, name):
        return float(getattr(node, name))
    if isinstance(node, dict) and name in node:
        return float(node[name])
    return float(default)


def _build_subrewards(cfg_reward, device: str) -> List[_SubReward]:
    """Build enabled sub-reward list from config. Weight 0 → skipped."""
    weights = cfg_reward.weights
    scales = cfg_reward.scales
    subs: List[_SubReward] = []

    registry = {
        "gradient": lambda: GradientReward(),
        "entropy": lambda: EntropyReward(),
        "eme": lambda: EMEReward(),
    }

    for name, ctor in registry.items():
        w = _get(weights, name, 0.0)
        if w == 0.0:
            continue
        subs.append(
            _SubReward(name=name, fn=ctor(), weight=w, scale=_get(scales, name, 1.0))
        )

    for iqa_name in ("clipiqa", "musiq", "niqe", "brisque"):
        w = _get(weights, iqa_name, 0.0)
        if w == 0.0:
            continue
        fn = _load_iqa(iqa_name, device)
        if fn is None:
            print(
                f"[reward] warn: pyiqa not available, skipping '{iqa_name}' "
                f"(weight={w}). Install pyiqa or set weight to 0."
            )
            continue
        subs.append(
            _SubReward(name=iqa_name, fn=fn, weight=w, scale=_get(scales, iqa_name, 1.0))
        )

    if not subs:
        raise ValueError(
            "CompositeReward has no active sub-rewards. Check `reward.weights` in config."
        )
    return subs


def _gaussian_kernel2d(radius: int, device, dtype) -> Tensor:
    """Returns a normalized 2D gaussian kernel of shape [1, 1, 2r+1, 2r+1]."""
    size = 2 * radius + 1
    sigma = max(radius / 2.0, 1e-3)
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    kernel = g1.view(-1, 1) * g1.view(1, -1)
    return kernel.view(1, 1, size, size)


class CompositeReward:
    """Weighted sum of sub-rewards, optionally relative, scalar or per-pixel."""

    def __init__(
        self,
        cfg_reward,
        device: str = "cpu",
    ) -> None:
        self.mode: str = str(cfg_reward.mode)
        # `terminal_borda` is GRPO-only and never calls into this object; we
        # still allow construction so the rest of the train script can wire
        # the env up uniformly. compute() will assert if invoked in that mode.
        assert self.mode in {"scalar", "pixel", "terminal_borda"}, f"invalid reward.mode: {self.mode}"
        self.relative: bool = bool(cfg_reward.relative)
        self.pixel_smooth_radius: int = int(cfg_reward.pixel_smooth_radius)
        self._device = device
        self._subs = _build_subrewards(cfg_reward, device=device)
        # Wrap with RelativeReward if requested; else leave absolute.
        if self.relative:
            for s in self._subs:
                s.fn = RelativeReward(s.fn)
        self._names = [s.name for s in self._subs]

    @property
    def active_names(self) -> List[str]:
        return list(self._names)

    def to(self, device) -> "CompositeReward":
        for s in self._subs:
            s.fn = s.fn.to(device)
        self._device = str(device)
        return self

    def _reduce_spatial_to_scalar(self, r: Tensor) -> Tensor:
        # r is [B, 1, H, W] -> [B]
        return r.mean(dim=(1, 2, 3))

    def _broadcast_scalar_to_spatial(self, r: Tensor, hw: Tuple[int, int]) -> Tensor:
        h, w = hw
        return r.view(-1, 1, 1, 1).expand(-1, 1, h, w).contiguous()

    def _smooth_pixel(self, r: Tensor) -> Tensor:
        if self.pixel_smooth_radius <= 0:
            return r
        k = _gaussian_kernel2d(self.pixel_smooth_radius, r.device, r.dtype)
        pad = self.pixel_smooth_radius
        return F.conv2d(r, k, padding=pad)

    def compute(
        self,
        x_prev: Tensor,
        x_curr: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return (reward, per-subreward means for logging).

        Reward shape depends on `self.mode`.
        """
        assert x_curr.ndim == 4
        assert self.mode in {"scalar", "pixel"}, (
            f"CompositeReward.compute() called in mode={self.mode!r}; "
            "terminal_borda must use the GRPO trainer's IQA-head path instead."
        )
        h, w = x_curr.shape[-2:]
        per_name_values: Dict[str, float] = {}

        if self.mode == "scalar":
            total = torch.zeros(x_curr.shape[0], device=x_curr.device, dtype=x_curr.dtype)
            for s in self._subs:
                r = s.fn.compute(x_prev, x_curr)  # [B] or [B,1,H,W]
                if r.ndim == 4:
                    r = self._reduce_spatial_to_scalar(r)
                r_scaled = s.scale * r
                total = total + s.weight * r_scaled
                per_name_values[s.name] = float(r_scaled.mean().detach().cpu())
            return total, per_name_values

        # pixel mode
        total_pix = torch.zeros(x_curr.shape[0], 1, h, w, device=x_curr.device, dtype=x_curr.dtype)
        for s in self._subs:
            r = s.fn.compute(x_prev, x_curr)
            if r.ndim == 1:
                r_pix = self._broadcast_scalar_to_spatial(r, (h, w))
            else:
                r_pix = r
            r_pix = self._smooth_pixel(r_pix)
            total_pix = total_pix + s.weight * s.scale * r_pix
            per_name_values[s.name] = float(r_pix.mean().detach().cpu())
        return total_pix, per_name_values
