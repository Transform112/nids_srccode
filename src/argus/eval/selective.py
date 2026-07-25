"""Selective prediction metrics: risk-coverage, AURC, deferral precision.

See docs/08_EVALUATION.md §4.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deferral_score: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the risk–coverage curve.

    Sorts samples by *deferral_score* (higher = more uncertain = defer first),
    then sweeps coverage from 100 % (defer nothing) to 0 % (defer everything).

    Args:
        y_true: [N] true labels.
        y_pred: [N] predicted labels.
        deferral_score: [N] per-sample uncertainty score (e.g. vacuity).
        n_points: resolution of the curve.

    Returns:
        coverage: [n_points] fraction of samples NOT deferred.
        risk: [n_points] error rate on non-deferred samples.
        thresholds: [n_points] deferral_score threshold at each point.
    """
    order = np.argsort(deferral_score)[::-1]  # highest uncertainty first
    y_pred_sorted = y_pred[order]
    y_true_sorted = y_true[order]

    coverage = np.linspace(1.0, 0.0, n_points)
    risk = np.zeros(n_points)
    thresholds = np.zeros(n_points)
    n = len(y_true)

    for i, cov in enumerate(coverage):
        n_keep = max(1, int(np.round(cov * n)))
        # Keep the n_keep lowest-uncertainty samples (end of sorted array)
        keep = order[-n_keep:] if n_keep > 0 else np.array([], dtype=np.int64)
        if len(keep) == 0:
            risk[i] = 0.0
            thresholds[i] = deferral_score[order[0]]
        else:
            risk[i] = float((y_true[keep] != y_pred[keep]).mean())
            if n_keep < n:
                thresholds[i] = deferral_score[order[-n_keep]]
            else:
                thresholds[i] = deferral_score[order[0]]
    return coverage, risk, thresholds


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area Under the Risk–Coverage curve. Lower is better."""
    return float(np.trapz(risk, coverage))


def e_aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Excess AURC over the optimal oracle ranking.

    The oracle always defers the highest-error samples first.
    """
    # Optimal: risk is monotonically decreasing from overall error to 0
    overall_error = risk[0]  # risk at coverage=1.0
    optimal_risk = overall_error * (1.0 - coverage)  # linear from overall_error to 0
    optimal_aurc = float(np.trapz(optimal_risk, coverage))
    return aurc(coverage, risk) - optimal_aurc


def risk_at_coverage(
    coverage: np.ndarray,
    risk: np.ndarray,
    target_coverage: float = 0.90,
) -> float:
    """Risk at a specific coverage level (interpolated)."""
    idx = np.argmin(np.abs(coverage - target_coverage))
    return float(risk[idx])


def deferral_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_deferred: np.ndarray,
) -> float:
    """Fraction of deferred flows that the non-deferring model would have got WRONG.

    A high value means DEFER is useful (it catches mistakes), not merely conservative.
    """
    if y_deferred.sum() == 0:
        return 1.0
    wrong = y_pred != y_true
    return float(wrong[y_deferred].sum() / y_deferred.sum())


def selective_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deferral_score: np.ndarray,
    n_points: int = 100,
) -> dict[str, Any]:
    """Full selective-prediction report.

    Args:
        y_true: [N] true labels.
        y_pred: [N] predicted labels.
        deferral_score: [N] per-sample uncertainty (higher → defer).
        n_points: curve resolution.

    Returns:
        Dict with aurc, e_aurc, risk_at_90, deferral_precision, and curve data.
    """
    cov, risk, thresholds = risk_coverage_curve(
        y_true, y_pred, deferral_score, n_points=n_points
    )
    # Deferral precision at the point where ~2% of flows are deferred (per config default)
    n_defer_target = max(1, int(len(y_true) * 0.02))
    idx_defer = np.argmin(np.abs(cov - (1.0 - 0.02)))
    thresh_2pct = thresholds[idx_defer]
    y_deferred = deferral_score > thresh_2pct

    return {
        "aurc": aurc(cov, risk),
        "e_aurc": e_aurc(cov, risk),
        "risk_at_90": risk_at_coverage(cov, risk, 0.90),
        "deferral_precision": deferral_precision(y_true, y_pred, y_deferred),
        "coverage": cov.tolist(),
        "risk": risk.tolist(),
    }
