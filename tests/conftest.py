"""Shared test fixtures."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# Ensure the project root is on sys.path so `import src....` works under pytest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
