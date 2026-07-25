"""I/O helpers: parquet/json/tensor load/save with consistent defaults.

See docs/12_IMPLEMENTATION_PLAN.md §4.3 — fail loudly on missing files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data as JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON, raising FileNotFoundError if absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a Parquet file, raising FileNotFoundError if absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Parquet file: {path}")
    return pd.read_parquet(path, columns=columns)


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a single tensor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str | Path, map_location: str = "cpu") -> torch.Tensor:
    """Load a single tensor from disk."""
    return torch.load(path, map_location=map_location, weights_only=True)


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk."""
    return np.load(path)
