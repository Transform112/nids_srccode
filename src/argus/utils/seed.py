"""Deterministic seeding for all RNGs from one run seed.

See docs/12_IMPLEMENTATION_PLAN.md §4.5. Where a non-deterministic kernel is
unavoidable, record it in env.json rather than silently accepting it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs from a single integer.

    Args:
        seed: Run-level seed from config.
        deterministic: If True, set CUDNN deterministic + benchmark=False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
