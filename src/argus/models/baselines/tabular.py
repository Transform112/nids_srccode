"""Flow-independent tabular baselines: Extra Trees, Random Forest, MLP.

See docs/05_ARCHITECTURE.md §8. These isolate whether the graph helps at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


@dataclass
class TabularBaseline:
    """Wraps a scikit-learn classifier trained on the flat F_e feature vector."""

    name: str
    model: Any

    @classmethod
    def extra_trees(cls, n_estimators: int = 200, seed: int = 0) -> "TabularBaseline":
        return cls(
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )

    @classmethod
    def random_forest(cls, n_estimators: int = 200, seed: int = 0) -> "TabularBaseline":
        return cls(
            "random_forest",
            RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )

    @classmethod
    def mlp(cls, hidden_sizes: tuple[int, ...] = (128, 64), seed: int = 0) -> "TabularBaseline":
        return cls(
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=hidden_sizes,
                random_state=seed,
                max_iter=200,
                early_stopping=True,
            ),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabularBaseline":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)
