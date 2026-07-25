"""Open-set recognition metrics.

See docs/08_EVALUATION.md §3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


def _to_binary(known_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert boolean known mask to binary labels: 1 = known, 0 = unknown."""
    return known_mask.astype(np.int32)


def unknown_tpr_fpr(y_known: np.ndarray, y_unknown_pred: np.ndarray) -> dict[str, float]:
    """Unknown detection TPR and FPR.

    Args:
        y_known: [N] bool — True if the sample belongs to a known class.
        y_unknown_pred: [N] bool — True if the model predicts UNKNOWN.

    Returns:
        Dict with ``unknown_tpr`` (fraction of TRULY unknown flows predicted
        UNKNOWN) and ``unknown_fpr`` (fraction of known flows wrongly predicted
        UNKNOWN).
    """
    truly_unknown = ~y_known
    if truly_unknown.sum() == 0:
        tpr = 0.0
    else:
        tpr = float((truly_unknown & y_unknown_pred).sum() / truly_unknown.sum())
    if y_known.sum() == 0:
        fpr = 0.0
    else:
        fpr = float((y_known & y_unknown_pred).sum() / y_known.sum())
    return {"unknown_tpr": tpr, "unknown_fpr": fpr}


def open_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    known_class_ids: set[int],
) -> float:
    """Joint measure of closed-set accuracy + unknown detection.

    Constructs a binary task: known (inlier, class ∈ known_class_ids) vs unknown
    (outlier). The unknown score is computed per sample from the vacuity or
    one minus the maximum known-class score.

    Args:
        y_true: [N] integer class labels.
        y_score: [N, C] class probabilities or evidence-based scores.
        known_class_ids: set of class indices that are "known".

    Returns:
        AUROC for the known-vs-unknown detection task (higher is better).
    """
    y_binary = np.array([1 if y in known_class_ids else 0 for y in y_true], dtype=np.int32)
    if len(np.unique(y_binary)) < 2:
        return 0.5  # degenerate — cannot compute AUROC
    # Unknown score: 1 - max known-class probability
    known_cols = sorted(known_class_ids)
    if len(known_cols) == y_score.shape[1]:
        # All classes are "known" — use 1 - max overall
        unknown_score = 1.0 - y_score.max(axis=1)
    else:
        unknown_score = 1.0 - y_score[:, known_cols].max(axis=1)
    # Known score is the inverse
    known_score = 1.0 - unknown_score
    # AUROC: higher = better separation. Use known_score for inliers.
    return float(roc_auc_score(y_binary, known_score))


def openness(raw_openness: float | None = None, c_train: int = 0, c_test: int = 0, c_target: int = 0) -> float:
    """Compute theoretical openness.

    ``openness = 1 - sqrt(2 * C_train / (C_test + C_target))``

    Args:
        raw_openness: if given, return as-is (already computed).
        c_train: number of classes seen during training.
        c_test: number of classes in the test set.
        c_target: number of classes the model is expected to recognise.

    Returns:
        Openness ∈ [0, 1].
    """
    if raw_openness is not None:
        return float(raw_openness)
    denom = c_test + c_target
    if denom == 0:
        return 1.0
    ratio = 2 * c_train / denom
    if ratio > 1.0:
        ratio = 1.0
    return 1.0 - np.sqrt(ratio)


def open_set_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    unknown_score: np.ndarray,
    known_class_ids: set[int],
    class_names: list[str],
    theta_unknown: float,
) -> dict[str, Any]:
    """Full open-set evaluation report for one holdout configuration.

    Args:
        y_true: [N] integer labels.
        y_pred: [N] integer predicted labels (known classes) or -1 for UNKNOWN.
        y_score: [N, C] class scores.
        unknown_score: [N] per-sample unknown score (e.g. vacuity u).
        known_class_ids: which class indices were known during training.
        class_names: all class names (known + unknown).
        theta_unknown: threshold on unknown_score for UNKNOWN prediction.

    Returns:
        Dict with all open-set metrics.
    """
    y_known = np.array([y in known_class_ids for y in y_true])
    y_unknown_pred = unknown_score > theta_unknown

    tpr_fpr = unknown_tpr_fpr(y_known, y_unknown_pred)

    # Known-class macro-F1 restricted to non-deferred flows
    from argus.eval.metrics import closed_set_report

    known_mask = ~y_unknown_pred
    known_names = [class_names[i] for i in sorted(known_class_ids)]

    if known_mask.sum() > 0 and len(known_class_ids) > 0:
        known_report = closed_set_report(
            y_true[known_mask], y_pred[known_mask], known_names
        )
    else:
        known_report = {"macro_f1": 0.0, "per_class_f1": {}}

    report: dict[str, Any] = {
        "unknown_tpr": tpr_fpr["unknown_tpr"],
        "unknown_fpr": tpr_fpr["unknown_fpr"],
        "open_auc": open_auc(y_true, y_score, known_class_ids),
        "known_macro_f1": known_report.get("macro_f1", 0.0),
        "theta_unknown": float(theta_unknown),
        "n_known_test": int(y_known.sum()),
        "n_unknown_test": int((~y_known).sum()),
        "n_deferred": int(y_unknown_pred.sum()),
    }
    return report
