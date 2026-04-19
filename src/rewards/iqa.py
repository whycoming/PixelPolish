"""No-reference IQA rewards via pyiqa (soft dependency).

If `pyiqa` is not installed, instantiating these classes raises a clear error;
the factory in `composite.py` checks availability and skips gracefully.
"""

from typing import Optional

import torch
from torch import Tensor

from src.rewards.base import RewardFunction


def _try_import_pyiqa():
    try:
        import pyiqa  # type: ignore

        return pyiqa
    except ImportError:
        return None


class _PyIQAWrappedReward(RewardFunction):
    """Base class for pyiqa-backed rewards. Frozen; called under no_grad."""

    spatial = False
    _metric_name: str = ""

    def __init__(self, device: str = "cpu") -> None:
        pyiqa = _try_import_pyiqa()
        if pyiqa is None:
            raise ImportError(
                f"{self.__class__.__name__} requires `pyiqa`. "
                "Install it (`pip install pyiqa`) or drop its weight to 0 in config."
            )
        self._metric = pyiqa.create_metric(self._metric_name, device=device, as_loss=False)
        # Some pyiqa metrics expect min-is-best (e.g., NIQE). We flip sign in subclasses as needed.
        self._higher_is_better = bool(getattr(self._metric, "lower_better", False)) is False
        self._device = device
        # Freeze all params.
        if hasattr(self._metric, "parameters"):
            for p in self._metric.parameters():
                p.requires_grad_(False)
        if hasattr(self._metric, "eval"):
            self._metric.eval()

    def to(self, device) -> "_PyIQAWrappedReward":
        self._device = str(device)
        try:
            self._metric.to(device)
        except Exception:
            pass
        return self

    def _compute(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            score = self._metric(x.to(self._device))
        if not torch.is_tensor(score):
            score = torch.tensor(score, device=x.device, dtype=x.dtype)
        score = score.to(x.device).float().view(-1)
        if not self._higher_is_better:
            score = -score
        return score


class CLIPIQAReward(_PyIQAWrappedReward):
    _metric_name = "clipiqa"


class MUSIQReward(_PyIQAWrappedReward):
    _metric_name = "musiq"


class NIQEReward(_PyIQAWrappedReward):
    _metric_name = "niqe"


class BRISQUEReward(_PyIQAWrappedReward):
    _metric_name = "brisque"


_HEAD_REGISTRY = {
    "clipiqa": CLIPIQAReward,
    "musiq": MUSIQReward,
    "niqe": NIQEReward,
    "brisque": BRISQUEReward,
}


def build_head(name: str, device: str = "cpu") -> Optional["_PyIQAWrappedReward"]:
    """Public factory for a single IQA head.

    Returns None if `pyiqa` is missing, the name is unknown, or instantiation
    fails (e.g., pretrained weights cannot be downloaded). Sign convention is
    already applied in `_compute` so higher = better in all cases.
    """
    if _try_import_pyiqa() is None:
        return None
    cls = _HEAD_REGISTRY.get(name)
    if cls is None:
        return None
    try:
        return cls(device=device)
    except Exception:
        return None
