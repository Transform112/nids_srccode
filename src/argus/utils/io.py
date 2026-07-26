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


def resolve_class_vocab(cfg: Any, dataset: str, artifact_dir: Path,
                        holdout_index: int | None = None) -> Path:
    """Resolve class_vocab.json, preferring the Stage-1 run directory.

    On Kaggle the artifact directory (input mount) is read-only, so
    ``04_train_encoder`` writes ``class_vocab.json`` to its run_dir.
    This helper checks that location first, then falls back to the
    artifact directory for local dev runs.

    Args:
        cfg: resolved OmegaConf config.
        dataset: dataset name (e.g. ``"cicids2018"``).
        artifact_dir: ``resolved_path(cfg, "artifact_dir") / dataset`` (already
            holdout-suffixed by the caller for Protocol-B runs).
        holdout_index: Protocol-B holdout index, if any — selects the
            matching ``<dataset>_stage1_b<i>`` run directory.

    Returns:
        Path to the class_vocab.json file that exists.
    """
    from pathlib import Path

    suffix = run_suffix(holdout_index)
    stage1_dir = Path(__file__).parents[3] / cfg.run.out_dir / f"{dataset}_stage1{suffix}"
    run_copy = stage1_dir / "class_vocab.json"
    artifact_copy = artifact_dir / "class_vocab.json"

    if run_copy.is_file():
        return run_copy
    if artifact_copy.is_file():
        return artifact_copy
    raise FileNotFoundError(
        f"class_vocab.json not found in {stage1_dir} or {artifact_dir}. "
        f"Run scripts/04_train_encoder.py first."
    )


def holdout_subdir(base: Path, holdout_index: int | None) -> Path:
    """Per-holdout data/artifact subdirectory for Protocol-B runs.

    Protocol-B open-set evaluation requires 5 genuinely holdout-excluded
    training runs (docs/02_DATASETS.md §4.1); each gets its own splits,
    feature pipeline, cache, and checkpoints under ``holdout_b<i>/`` so they
    never clobber the Protocol-A closed-set pipeline or each other.
    """
    return base if holdout_index is None else base / f"holdout_b{holdout_index}"


def run_suffix(holdout_index: int | None) -> str:
    """Run-directory / run-id suffix for Protocol-B holdout runs ("_b<i>")."""
    return "" if holdout_index is None else f"_b{holdout_index}"


def derive_class_vocab(train_df: pd.DataFrame) -> list[str]:
    """Derive the canonical class vocabulary directly from train-split labels.

    This is the single source of truth for label->id ordering. Scripts
    03_cache_graphs.py and 04_train_encoder.py must both call this (rather
    than one reading a persisted class_vocab.json) because 03 runs *before*
    04 writes that file in the intended Kaggle workflow — a CPU-only
    cache-building session ahead of the GPU training session. Deriving
    fresh from the same train_features.parquet guarantees identical label
    ids regardless of run order.
    """
    return sorted(train_df["canonical_label"].unique().tolist())
