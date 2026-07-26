"""Checkpoint save/resume with full RNG state.

See docs/06_TRAINING.md §6.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer/RNG state for resumable training.

    Written atomically (temp file + rename) so a session killed mid-save
    never corrupts the only resumable checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore model/optimizer/RNG state."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state
