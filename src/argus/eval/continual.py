"""Continual-learning metrics: forgetting, backward transfer, few-shot.

See docs/08_EVALUATION.md §4.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def forgetting(
    per_class_f1_before: dict[str, float],
    per_class_f1_after: dict[str, float],
) -> float:
    """Maximum drop in per-class F1 after registering new classes.

    ``forgetting_c = max(0, F1_before(c) − F1_after(c))`` across all old classes.
    ARGUS must return exactly 0.000 by construction.

    Returns:
        Maximum forgetting across all classes.
    """
    drops = []
    for c, f1_before in per_class_f1_before.items():
        f1_after = per_class_f1_after.get(c, f1_before)
        drops.append(max(0.0, f1_before - f1_after))
    return float(max(drops)) if drops else 0.0


def backward_transfer(
    per_class_f1_before: dict[str, float],
    per_class_f1_after: dict[str, float],
) -> float:
    """Backward transfer: mean change in F1 on old classes.

    Positive BWT means old classes improved after learning new ones (unusual).
    Negative BWT means forgetting (typical for fine-tuning).
    Zero BWT is the ARGUS guarantee.

    ``BWT = mean_c ( F1_after(c) − F1_before(c) )`` over old classes.
    """
    deltas = []
    for c, f1_before in per_class_f1_before.items():
        f1_after = per_class_f1_after.get(c, f1_before)
        deltas.append(f1_after - f1_before)
    return float(np.mean(deltas)) if deltas else 0.0


def few_shot_report(
    new_class_f1: dict[int, float],
    old_class_f1_before: dict[str, float],
    old_class_f1_after: dict[str, float],
    n_shots: int,
    registration_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Report for a single few-shot registration run.

    Args:
        new_class_f1: per-registered-class F1, keyed by n_shots.
        old_class_f1_before: per-class F1 before registration.
        old_class_f1_after: per-class F1 after registration.
        n_shots: number of labelled samples used for registration.
        registration_latency_ms: wall-clock registration time.

    Returns:
        Dict with all few-shot metrics (docs/08_EVALUATION.md §4.3).
    """
    macro_before = float(np.mean(list(old_class_f1_before.values())))
    macro_after = float(np.mean(list(old_class_f1_after.values())))

    return {
        "n_shots": n_shots,
        "new_class_f1": new_class_f1,
        "old_class_macro_f1_before": macro_before,
        "old_class_macro_f1_after": macro_after,
        "old_class_macro_f1_delta": macro_after - macro_before,
        "forgetting": forgetting(old_class_f1_before, old_class_f1_after),
        "bwt": backward_transfer(old_class_f1_before, old_class_f1_after),
        "registration_latency_ms": registration_latency_ms,
    }
