"""Unpaired and mixed-modality image datasets.

`UnpairedImageDataset` recursively scans a directory for image files and
returns tensors in `[0, 1]` of shape `[C, H, W]`. Grayscale images are
broadcast to `channels` channels for modality-agnostic policy input.

`MixedModalityDataset` samples balanced batches from multiple sources (e.g.
RGB low-light + grayscale + IR) so that `batch_size` iterations draw roughly
equal counts from each modality regardless of dataset size.
"""

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import random

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _scan(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTS and p.is_file())


class UnpairedImageDataset(Dataset):
    """Directory of images → float tensors in [0, 1]."""

    def __init__(self, root: str, image_size: int, channels: int = 3) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.paths = _scan(self.root)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")
        self._tx = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path)
        if img.mode == "L":
            img = img.convert("L")
            x = self._tx(img)
            if self.channels > 1:
                x = x.expand(self.channels, -1, -1).contiguous()
        else:
            img = img.convert("RGB")
            x = self._tx(img)
            if self.channels == 1:
                x = x.mean(dim=0, keepdim=True)
        assert x.shape == (self.channels, self.image_size, self.image_size), (
            f"unexpected shape {tuple(x.shape)} for {path}"
        )
        return x


class MixedModalityDataset(Dataset):
    """Weighted mixture of directories with balanced sampling.

    `modalities` is a sequence of `(weight, root_path)` pairs. On each __getitem__
    a modality is sampled according to weights, then a random image within that
    modality is drawn. `__len__` is the sum of sub-dataset lengths (used by
    DataLoader for scheduling; actual sampling is random per-iteration).
    """

    def __init__(
        self,
        modalities: Sequence[Tuple[float, str]],
        image_size: int,
        channels: int = 3,
        rng_seed: int = 0,
    ) -> None:
        if not modalities:
            raise ValueError("MixedModalityDataset requires at least one (weight, path) pair.")
        self._subs = [
            UnpairedImageDataset(root, image_size=image_size, channels=channels)
            for _, root in modalities
        ]
        self._weights = torch.tensor([float(w) for w, _ in modalities], dtype=torch.float32)
        assert torch.all(self._weights > 0), "modality weights must be positive"
        self._weights = self._weights / self._weights.sum()
        self._rng = random.Random(rng_seed)
        self._total = sum(len(s) for s in self._subs)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tensor:
        modality_idx = int(torch.multinomial(self._weights, num_samples=1).item())
        sub = self._subs[modality_idx]
        inner = self._rng.randrange(len(sub))
        return sub[inner]
