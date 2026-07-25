"""Identity-only classifier: the leakage floor, not a competitor.

Trained on (src IP, dst IP, src port bucket, dst port bucket) only. Quantifies
how much of a headline number could be identity memorisation rather than
behavioural learning. See docs/02_DATASETS.md §7.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _port_bucket(p: pd.Series) -> pd.Series:
    return pd.cut(
        p, bins=[-1, 1023, 49151, 65535], labels=["well_known", "registered", "ephemeral"]
    ).astype(str)


class IdentityOnlyClassifier:
    """Depth-8 decision tree on identity-only features. See docs/02_DATASETS.md §7.1."""

    def __init__(self, max_depth: int = 8, seed: int = 0) -> None:
        self.max_depth = max_depth
        self.seed = seed
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        self._src_categories: pd.Index | None = None
        self._dst_categories: pd.Index | None = None

    def _encode(
        self,
        df: pd.DataFrame,
        src_col: str = "IPV4_SRC_ADDR",
        dst_col: str = "IPV4_DST_ADDR",
        src_port_col: str = "L4_SRC_PORT",
        dst_port_col: str = "L4_DST_PORT",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "src_ip": pd.Categorical(df[src_col], categories=self._src_categories).codes,
                "dst_ip": pd.Categorical(df[dst_col], categories=self._dst_categories).codes,
                "src_port_bucket": _port_bucket(df[src_port_col]).astype("category").cat.codes,
                "dst_port_bucket": _port_bucket(df[dst_port_col]).astype("category").cat.codes,
            }
        )

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, test_df: pd.DataFrame | None = None) -> "IdentityOnlyClassifier":
        """Fit on train; if test_df given, build the shared category vocabulary
        across train+test so unseen IPs at test time map to a valid (missing)
        code rather than raising.
        """
        union_src = train_df["IPV4_SRC_ADDR"]
        union_dst = train_df["IPV4_DST_ADDR"]
        if test_df is not None:
            union_src = pd.concat([union_src, test_df["IPV4_SRC_ADDR"]])
            union_dst = pd.concat([union_dst, test_df["IPV4_DST_ADDR"]])
        self._src_categories = pd.Categorical(union_src).categories
        self._dst_categories = pd.Categorical(union_dst).categories

        x = self._encode(train_df)
        self.model.fit(x, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._encode(df))
