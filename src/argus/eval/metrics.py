"""Evaluation metrics for closed-set classification.

See docs/08_EVALUATION.md §2. Accuracy is never the headline metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
)


def closed_set_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the full closed-set metric suite.

    Args:
        y_true: [N] integer class labels
        y_pred: [N] integer predicted labels
        class_names: ordered class name list (index -> name)
        y_score: optional [N, C] class probabilities for PR-AUC
    Returns:
        dict with macro_f1, per_class_f1, weighted_f1, mcc, balanced_accuracy,
        confusion_matrix, and (if y_score given) pr_auc per class.
    """
    labels = list(range(len(class_names)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    report: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": {name: float(v) for name, v in zip(class_names, per_class_f1)},
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "support": {name: int((y_true == i).sum()) for i, name in enumerate(class_names)},
    }
    if y_score is not None:
        pr_auc = {}
        for i, name in enumerate(class_names):
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() == 0:
                pr_auc[name] = None
                continue
            precision, recall, _ = precision_recall_curve(binary_true, y_score[:, i])
            pr_auc[name] = float(auc(recall, precision))
        report["pr_auc"] = pr_auc
    return report


def per_tier_macro_f1(
    per_class_f1: dict[str, float],
    support: dict[str, int],
    head_threshold: int = 1_000_000,
    body_threshold: int = 50_000,
    tail_threshold: int = 1_000,
) -> dict[str, float]:
    """Aggregate per-class F1 into head/body/tail/extreme tiers by support.

    See docs/02_DATASETS.md §6.1 and docs/08_EVALUATION.md §2.
    """
    tiers: dict[str, list[float]] = {"head": [], "body": [], "tail": [], "extreme": []}
    for name, f1 in per_class_f1.items():
        n = support.get(name, 0)
        if n > head_threshold:
            tiers["head"].append(f1)
        elif n > body_threshold:
            tiers["body"].append(f1)
        elif n > tail_threshold:
            tiers["tail"].append(f1)
        else:
            tiers["extreme"].append(f1)
    return {k: float(np.mean(v)) if v else None for k, v in tiers.items()}
