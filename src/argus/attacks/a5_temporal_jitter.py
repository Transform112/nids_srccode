"""A5 — temporal jitter.

See docs/10_ADVERSARIAL.md §6. Perturbs the attacker's own inter-packet
arrival statistics with multiplicative log-normal jitter, recomputes TE2
derived features via the frozen `FeaturePipeline` (never perturbs derived
features directly), and reports both attack success and **attacker cost**.

Attacker-cost approximation: only aggregate IAT statistics are available at
this dataset's flow granularity (not per-packet timestamps), so
`FLOW_DURATION_MILLISECONDS` and `DURATION_IN`/`DURATION_OUT` are scaled by
the mean of the two directions' jitter factors — consistent with the physical
fact that inter-packet gaps compose additively into flow duration. This is an
explicit, documented approximation, not a claim of exact packet-level
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from argus.attacks.constraints import project
from argus.attacks.raw_evasion_common import evaluate_raw_row
from argus.models.argus import ArgusModel

IAT_PREFIXES = ("SRC_TO_DST_IAT", "DST_TO_SRC_IAT")
IAT_SUFFIXES = ("_MIN", "_AVG", "_MAX", "_STDDEV")
DURATION_COLUMNS = ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT")


@dataclass
class JitterResult:
    sigma: float
    clean_decision: int
    attacked_decision: int
    clean_evidence_total: float
    attacked_evidence_total: float
    evaded: bool
    duration_change_frac: float  # attacker cost: induced change in flow duration
    pkt_rate_change_frac: float  # attacker cost: induced change in effective packet rate


def apply_temporal_jitter(
    raw_row: pd.DataFrame, sigma: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, float]:
    """Multiplicatively jitter each direction's IAT stats by an independent
    log-normal factor, scale duration by the mean factor, then project onto
    domain constraints. Returns (jittered_row, mean_jitter_factor).
    """
    row = raw_row.copy().reset_index(drop=True)
    factors = []
    for prefix in IAT_PREFIXES:
        avg_col = f"{prefix}_AVG"
        if avg_col not in row.columns:
            continue
        factor = float(np.exp(rng.normal(0.0, sigma)))
        factors.append(factor)
        for suffix in IAT_SUFFIXES:
            col = f"{prefix}{suffix}"
            if col in row.columns:
                row[col] = row[col] * factor

    mean_factor = float(np.mean(factors)) if factors else 1.0
    for col in DURATION_COLUMNS:
        if col in row.columns:
            row[col] = row[col] * mean_factor

    row = project(row)
    return row, mean_factor


def run_a5_jitter_sweep(
    model: ArgusModel,
    pipeline,
    batch: dict,
    raw_row: pd.DataFrame,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
    sigmas: list[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
    benign_class_id: int = 0,
    seed: int = 0,
) -> list[JitterResult]:
    rng = np.random.default_rng(seed)
    raw_row = raw_row.reset_index(drop=True)

    clean_decision, clean_evidence, _ = evaluate_raw_row(
        model, batch, raw_row, pipeline, feature_names, target_index_within_bin, device
    )

    results = []
    for sigma in sigmas:
        if sigma == 0.0:
            jittered, factor = raw_row, 1.0
        else:
            jittered, factor = apply_temporal_jitter(raw_row, sigma, rng)

        decision, evidence, _ = evaluate_raw_row(
            model, batch, jittered, pipeline, feature_names, target_index_within_bin, device
        )
        duration_change = factor - 1.0
        pkt_rate_change = (1.0 / factor - 1.0) if factor > 0 else 0.0  # rate ~ 1/duration, fixed pkt count
        results.append(
            JitterResult(
                sigma=sigma,
                clean_decision=clean_decision,
                attacked_decision=decision,
                clean_evidence_total=clean_evidence,
                attacked_evidence_total=evidence,
                evaded=(decision == benign_class_id),
                duration_change_frac=duration_change,
                pkt_rate_change_frac=pkt_rate_change,
            )
        )
    return results
