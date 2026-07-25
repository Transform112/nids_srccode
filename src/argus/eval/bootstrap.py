"""Paired bootstrap confidence intervals for model comparisons.

See docs/08_EVALUATION.md §7: paired comparisons use a paired bootstrap over
test flows, 10,000 resamples, reporting the 95% CI of the difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import resample


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute paired bootstrap confidence interval for the difference in a metric.

    Resamples test *indices* (the same indices for both models), computes the
    metric for each, then builds the CI of the difference `metric(a) − metric(b)`.

    Args:
        scores_a: per-sample scores from model A (e.g. logits, probabilities).
        scores_b: per-sample scores from model B, aligned to the same samples.
        metric_fn: callable ``(scores, y_true) -> float``.
        n_resamples: number of bootstrap draws.
        ci: confidence interval width (0.95 → 95% CI).
        seed: random seed for reproducibility.

    Returns:
        Dict with ``mean_diff``, ``ci_lower``, ``ci_upper``, ``ci_level``,
        ``n_resamples``, and ``significant`` (True if CI excludes 0).
    """
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = np.zeros(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        idx = resample(np.arange(n), n_samples=n, random_state=rng, replace=True)
        ma = metric_fn(scores_a[idx])
        mb = metric_fn(scores_b[idx])
        diffs[i] = ma - mb

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(diffs, alpha))
    upper = float(np.quantile(diffs, 1.0 - alpha))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "n_resamples": n_resamples,
        "significant": lower > 0 or upper < 0,  # CI excludes zero
    }


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Simple bootstrap CI for a single array of scalar metric values.

    This is the non-paired variant — use when comparing across seeds or holdouts
    rather than within a single test set.

    Args:
        values: [N] per-run metric values (e.g. macro-F1 from 5 seeds).
        n_resamples: bootstrap draws.
        ci: confidence interval width.
        seed: random seed.

    Returns:
        Dict with ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
    """
    rng = np.random.default_rng(seed)
    means = np.zeros(n_resamples, dtype=np.float64)
    n = len(values)
    for i in range(n_resamples):
        idx = resample(np.arange(n), n_samples=n, random_state=rng, replace=True)
        means[i] = float(np.mean(values[idx]))

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
    }
