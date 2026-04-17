"""Checkpoint save/load with simple retention policy."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model + optimizer + metadata to `path`."""
    obj: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if extra is not None:
        obj["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load model (required) and optimizer (optional) state. Returns the raw dict."""
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"])
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    return obj


def prune_old_checkpoints(directory: str, keep_last: int) -> None:
    """Keep only the most recent `keep_last` *.pt files in `directory` (sorted by mtime)."""
    d = Path(directory)
    if not d.exists():
        return
    ckpts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last] if keep_last > 0 else ckpts[:-1]:
        try:
            old.unlink()
        except OSError:
            pass
