from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import MixedModalityDataset, UnpairedImageDataset


def _write_rgb(path: Path, size: int = 64) -> None:
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_gray(path: Path, size: int = 64) -> None:
    arr = (np.random.rand(size, size) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_dataset_rgb_unit_range(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "a.png")
    _write_rgb(tmp_path / "b.jpg")
    ds = UnpairedImageDataset(str(tmp_path), image_size=32, channels=3)
    assert len(ds) == 2
    x = ds[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_dataset_broadcasts_grayscale(tmp_path: Path) -> None:
    _write_gray(tmp_path / "g.png")
    ds = UnpairedImageDataset(str(tmp_path), image_size=32, channels=3)
    x = ds[0]
    assert x.shape == (3, 32, 32)
    assert torch.allclose(x[0], x[1])
    assert torch.allclose(x[1], x[2])


def test_dataset_collapse_rgb_to_gray(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "a.png")
    ds = UnpairedImageDataset(str(tmp_path), image_size=32, channels=1)
    x = ds[0]
    assert x.shape == (1, 32, 32)


def test_dataset_raises_on_empty(tmp_path: Path) -> None:
    try:
        UnpairedImageDataset(str(tmp_path), image_size=32, channels=3)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_mixed_modality_dataset_samples(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    gray_dir = tmp_path / "gray"
    rgb_dir.mkdir(); gray_dir.mkdir()
    for i in range(3):
        _write_rgb(rgb_dir / f"r{i}.png")
    for i in range(2):
        _write_gray(gray_dir / f"g{i}.png")
    ds = MixedModalityDataset(
        modalities=[(1.0, str(rgb_dir)), (1.0, str(gray_dir))],
        image_size=32, channels=3,
    )
    assert len(ds) == 5
    for _ in range(10):
        x = ds[0]
        assert x.shape == (3, 32, 32)
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
