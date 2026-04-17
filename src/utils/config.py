"""Config loading with optional override file(s)."""

from pathlib import Path
from typing import Iterable, Optional, Union

from omegaconf import DictConfig, OmegaConf


PathLike = Union[str, Path]


def load_config(
    base_path: PathLike,
    overrides: Optional[Union[PathLike, Iterable[PathLike]]] = None,
    cli_overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    """Load a YAML config, optionally merging one or more override files and CLI dotlist overrides.

    Args:
        base_path: path to base YAML.
        overrides: single path or iterable of paths. Each is loaded and merged
            on top of the base in order. Keys in later files replace earlier ones.
        cli_overrides: iterable of "key.path=value" strings (OmegaConf dotlist).

    Returns:
        OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(str(base_path))
    if overrides is not None:
        if isinstance(overrides, (str, Path)):
            overrides = [overrides]
        for path in overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(str(path)))
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
    assert isinstance(cfg, DictConfig)
    return cfg


def resolve_device(name: str) -> str:
    """Return `cpu` if CUDA unavailable, else the requested device name."""
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name
