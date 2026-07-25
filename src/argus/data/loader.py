"""Streaming CSV loader for NF-v3 datasets.

Reads in chunks to bound memory on a laptop; never loads a 20M-row CSV whole
unless explicitly requested (e.g. small dev subsample already on disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from argus.constants import LABEL_COLS, RAW_FEATURE_COLS


def read_csv_chunks(
    csv_path: str | Path,
    chunksize: int = 500_000,
    nrows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a raw NF-v3 CSV.

    Args:
        csv_path: path to the raw CSV.
        chunksize: rows per chunk.
        nrows: optional cap on total rows read (for laptop dev runs).
    """
    usecols = RAW_FEATURE_COLS + LABEL_COLS
    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols,
        chunksize=chunksize,
        nrows=nrows,
    )
    for chunk in reader:
        yield chunk


def load_full(csv_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load an entire CSV (or first `nrows`) into memory. Use only for small files."""
    usecols = RAW_FEATURE_COLS + LABEL_COLS
    return pd.read_csv(csv_path, usecols=lambda c: c in usecols, nrows=nrows)
