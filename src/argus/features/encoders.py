"""Categorical, bitfield, and port encoders.

See docs/03_FEATURE_ENGINEERING.md §5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argus.constants import TCP_FLAG_BITS, TCP_FLAG_COLS


def expand_tcp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Expand TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS to 8 bits each (24 dims)."""
    out = pd.DataFrame(index=df.index)
    for col in TCP_FLAG_COLS:
        values = df[col].fillna(0).astype(int).to_numpy()
        for i, bit in enumerate(TCP_FLAG_BITS):
            out[f"{col}_{bit}"] = ((values >> i) & 1).astype(float)
    return out


def port_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """4 features per port column: well_known/registered/ephemeral flags + log."""
    p = df[col].fillna(0).astype(float)
    out = pd.DataFrame(index=df.index)
    out[f"{col}_is_well_known"] = (p < 1024).astype(float)
    out[f"{col}_is_registered"] = ((p >= 1024) & (p < 49152)).astype(float)
    out[f"{col}_is_ephemeral"] = (p >= 49152).astype(float)
    out[f"{col}_log"] = np.log1p(p) / np.log1p(65535)
    return out


class TopKOneHotEncoder:
    """Fit-on-train top-k category encoder with an OTHER bucket."""

    def __init__(self, column: str, k: int, prefix: str | None = None) -> None:
        self.column = column
        self.k = k
        self.prefix = prefix or column
        self.categories_: list = []

    def fit(self, df: pd.DataFrame) -> "TopKOneHotEncoder":
        counts = df[self.column].value_counts()
        self.categories_ = counts.head(self.k).index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        values = df[self.column]
        for cat in self.categories_:
            out[f"{self.prefix}_{cat}"] = (values == cat).astype(float)
        other_rate = (~values.isin(self.categories_)).mean()
        out[f"{self.prefix}_OTHER"] = (~values.isin(self.categories_)).astype(float)
        self.last_other_rate_ = float(other_rate)
        return out


def packet_size_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise packet-size histogram bins to a simplex + log row-sum."""
    from argus.constants import PKT_SIZE_BINS

    bins = df[PKT_SIZE_BINS].astype(float)
    row_sum = bins.sum(axis=1)
    out = bins.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)
    out.columns = [f"pkt_size_hist_norm_{c}" for c in PKT_SIZE_BINS]
    out["pkt_size_hist_sum_log"] = np.log1p(row_sum)
    return out
