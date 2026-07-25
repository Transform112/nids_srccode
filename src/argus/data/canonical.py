"""Label canonicalisation applied to a DataFrame.

See docs/02_DATASETS.md §5.2.
"""

from __future__ import annotations

import pandas as pd

from argus.constants import canonicalise


def canonicalise_labels(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add a `canonical_label` column, raising on unrecognised raw labels."""
    df = df.copy()
    df["canonical_label"] = df["Attack"].map(lambda raw: canonicalise(dataset, raw))
    return df
