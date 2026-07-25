"""TE1 — heavy-tail conditioning: signed-log + quantile transform.

See docs/03_FEATURE_ENGINEERING.md §3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from argus.constants import TE1_COLUMNS


class TE1Conditioner:
    """Fit-on-train-only signed-log + quantile-to-normal conditioning."""

    def __init__(
        self,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000,
        clip: float = 5.0,
        columns: list[str] | None = None,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.clip = clip
        self.columns = columns or list(TE1_COLUMNS)
        self.transformer: QuantileTransformer | None = None

    @staticmethod
    def _signed_log(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, None)
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, df: pd.DataFrame) -> "TE1Conditioner":
        present = [c for c in self.columns if c in df.columns]
        self.columns = present
        x = self._signed_log(df[present].to_numpy(dtype=float))
        self.transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(self.n_quantiles, max(len(df), 2)),
            subsample=self.subsample,
        )
        self.transformer.fit(x)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None:
            raise RuntimeError("TE1Conditioner must be fit before transform")
        x = self._signed_log(df[self.columns].to_numpy(dtype=float))
        x = self.transformer.transform(x)
        x = np.clip(x, -self.clip, self.clip)
        return pd.DataFrame(x, columns=self.columns, index=df.index)
