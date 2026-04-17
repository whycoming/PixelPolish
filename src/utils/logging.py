"""Minimal logger with TensorBoard backend and a no-op fallback."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor


class Logger:
    """Wraps tensorboard SummaryWriter; `backend='none'` disables I/O."""

    def __init__(self, log_dir: str, backend: str = "tensorboard") -> None:
        self.log_dir = Path(log_dir)
        self.backend = backend
        self._writer: Optional[Any] = None
        if backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.log_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(str(self.log_dir))
            except Exception as exc:  # pragma: no cover
                print(f"[Logger] TensorBoard unavailable ({exc}); falling back to stdout.")
                self.backend = "stdout"
        elif backend == "stdout":
            pass
        elif backend != "none":
            raise ValueError(f"Unknown logging backend: {backend}")

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        if self.backend == "none":
            return
        if self._writer is not None:
            for k, v in scalars.items():
                self._writer.add_scalar(k, float(v), step)
        if self.backend == "stdout":
            pretty = " ".join(f"{k}={float(v):.4f}" for k, v in scalars.items())
            print(f"[step {step}] {pretty}")

    def log_images(self, tag: str, images: Tensor, step: int) -> None:
        """Log a batch of images [B, C, H, W] in [0, 1] as a grid."""
        if self.backend == "none" or self._writer is None:
            return
        from torchvision.utils import make_grid

        grid = make_grid(images.clamp(0.0, 1.0).cpu(), nrow=min(4, images.size(0)))
        self._writer.add_image(tag, grid, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
