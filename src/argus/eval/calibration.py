"""Calibration metrics: ECE, MCE, Brier, NLL, reliability diagram data.

See docs/08_EVALUATION.md §4.1. Uses equal-mass bins per the spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss


def _equal_mass_bins(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition samples into ``n_bins`` equal-mass bins by confidence.

    Returns:
        bin_conf: mean confidence per bin.
        bin_acc: mean accuracy per bin.
        bin_weights: fraction of samples per bin.
    """
    order = np.argsort(confidences)
    n = len(confidences)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_weights = np.zeros(n_bins)
    for b in range(n_bins):
        start = int(np.round(b * n / n_bins))
        end = int(np.round((b + 1) * n / n_bins))
        if start == end:
            bin_conf[b] = 0.0
            bin_acc[b] = 0.0
            bin_weights[b] = 0.0
            continue
        idx = order[start:end]
        bin_conf[b] = float(np.mean(confidences[idx]))
        bin_acc[b] = float(np.mean(accuracies[idx]))
        bin_weights[b] = len(idx) / n
    return bin_conf, bin_acc, bin_weights


def ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_mass",
) -> float:
    """Expected Calibration Error.

    ``ECE = Σ_b (n_b/N) · |acc(b) − conf(b)|``

    Args:
        y_true: [N] integer labels.
        y_prob: [N, C] softmax probabilities.
        n_bins: number of bins.
        binning: ``"equal_mass"`` (default) or ``"equal_width"``.

    Returns:
        ECE ∈ [0, 1].
    """
    confidences = y_prob.max(axis=1)
    accuracies = (y_prob.argmax(axis=1) == y_true).astype(np.float64)

    if binning == "equal_mass":
        b_conf, b_acc, b_weight = _equal_mass_bins(confidences, accuracies, n_bins)
    else:
        # equal_width
        bin_edges = np.linspace(0, 1, n_bins + 1)
        b_conf = np.zeros(n_bins)
        b_acc = np.zeros(n_bins)
        b_weight = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            b_conf[i] = float(np.mean(confidences[mask]))
            b_acc[i] = float(np.mean(accuracies[mask]))
            b_weight[i] = mask.sum() / len(confidences)

    return float(np.sum(b_weight * np.abs(b_acc - b_conf)))


def mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_mass",
) -> float:
    """Maximum Calibration Error — the worst-bin calibration gap."""
    confidences = y_prob.max(axis=1)
    accuracies = (y_prob.argmax(axis=1) == y_true).astype(np.float64)

    if binning == "equal_mass":
        b_conf, b_acc, _ = _equal_mass_bins(confidences, accuracies, n_bins)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        b_conf = np.zeros(n_bins)
        b_acc = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            b_conf[i] = float(np.mean(confidences[mask]))
            b_acc[i] = float(np.mean(accuracies[mask]))
    return float(np.max(np.abs(b_acc - b_conf)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multi-class Brier score.

    ``BS = (1/N) Σ_i Σ_c (p_{i,c} − 1[y_i=c])²``
    """
    n, c = y_prob.shape
    y_onehot = np.zeros((n, c), dtype=np.float64)
    y_onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Negative log-likelihood on probabilities."""
    return float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1]))))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Return data for a reliability diagram plot.

    Returns:
        Dict with ``bin_centers``, ``bin_accuracy``, ``bin_count`` lists.
    """
    confidences = y_prob.max(axis=1)
    accuracies = (y_prob.argmax(axis=1) == y_true).astype(np.float64)
    b_conf, b_acc, b_weight = _equal_mass_bins(confidences, accuracies, n_bins)
    n = len(confidences)
    return {
        "bin_confidence": b_conf.tolist(),
        "bin_accuracy": b_acc.tolist(),
        "bin_count": [int(np.round(w * n)) for w in b_weight],
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_mass",
) -> dict[str, Any]:
    """Full calibration suite.

    Returns dict with ece, mce, brier, nll, and reliability diagram data.
    """
    return {
        "ece": ece(y_true, y_prob, n_bins=n_bins, binning=binning),
        "mce": mce(y_true, y_prob, n_bins=n_bins, binning=binning),
        "brier": brier_score(y_true, y_prob),
        "nll": nll(y_true, y_prob),
        "n_bins": n_bins,
        "binning": binning,
        "reliability": reliability_diagram_data(y_true, y_prob, n_bins=n_bins),
    }
