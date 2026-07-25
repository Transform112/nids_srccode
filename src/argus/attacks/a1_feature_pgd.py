"""A1 — constrained feature-space evasion.

See docs/10_ADVERSARIAL.md §2. Restricted to Channel A (attacker-controllable)
**raw** numeric columns of the target flow's own record.

The doc's pseudocode assumes gradients flow through `pipeline.transform`. In
this codebase the pipeline includes a quantile transform and one-hot encoders
(`sklearn`, non-autograd-traceable) — see `docs/10_ADVERSARIAL.md` §1 for why
attacks must operate in raw space at all. We therefore estimate the ascent
direction with **SPSA** (simultaneous perturbation stochastic approximation):
two forward passes per step regardless of the number of perturbed columns,
rather than one directional derivative per column. This is a standard
black-box gradient estimator and is a practical substitute for backprop
through non-differentiable preprocessing — the same kind of pragmatic
substitution already used for A2's KDE sampler.

TE2 derived features are recomputed by `pipeline.transform` at every step —
never perturbed directly (mandatory per `docs/TODO.md` standing rule 8).

Budget `epsilon` is defined as a **relative** (fractional) change on each raw
controllable column, e.g. `epsilon=0.2` allows each column to move up to ±20%
of its clean value. This is a deliberate, documented reinterpretation of the
doc's "epsilon in normalised units" — raw byte/packet counts have wildly
different scales, so a relative raw-space budget is the more meaningful and
auditable quantity for a security audience, and it composes correctly with
`constraints.project`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from argus.attacks.constraints import project
from argus.attacks.raw_evasion_common import evaluate_raw_row
from argus.models.argus import ArgusModel

# Raw, purely-numeric Channel A columns that a PGD-style continuous
# perturbation can act on. Categorical Channel A columns (protocol, ports,
# TCP flag bits) are left fixed — perturbing them is a discrete decision, not
# a continuous one, and is out of scope for this attack (see A4 for a
# white-box attacker that may exploit those too).
RAW_NUMERIC_CONTROLLABLE = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
]


@dataclass
class A1Result:
    epsilon: float
    clean_decision: int
    attacked_decision: int
    clean_evidence_total: float
    attacked_evidence_total: float
    evaded: bool
    steps_run: int


def _benign_neg_log_prob(p_hat_row: torch.Tensor, benign_class_id: int) -> float:
    """Loss to MINIMISE: -log p(benign). Lower loss = more benign-like."""
    return float(-torch.log(p_hat_row[benign_class_id].clamp_min(1e-12)))


def run_a1_pgd(
    model: ArgusModel,
    pipeline,
    batch: dict,
    raw_row: pd.DataFrame,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
    epsilon: float,
    steps: int = 40,
    spsa_c: float = 0.1,
    lr_frac: float = 0.15,
    benign_class_id: int = 0,
    seed: int = 0,
) -> A1Result:
    """Run one SPSA-PGD attack on a single target flow at a fixed `epsilon` budget.

    Args:
        batch: a pre-built bin batch (context unchanged throughout the attack).
        raw_row: 1-row DataFrame of the target flow's RAW (pre-pipeline) columns.
        feature_names: transformed feature column order (`pipeline.feature_names_`).
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in RAW_NUMERIC_CONTROLLABLE if c in raw_row.columns]

    x0 = raw_row.copy().reset_index(drop=True)
    if cols:
        x0[cols] = x0[cols].astype(np.float64)
    lo_bound = x0[cols].to_numpy(dtype=float) * (1.0 - epsilon)
    hi_bound = x0[cols].to_numpy(dtype=float) * (1.0 + epsilon)

    clean_decision, clean_evidence, _ = evaluate_raw_row(
        model, batch, x0, pipeline, feature_names, target_index_within_bin, device
    )

    if epsilon <= 0.0 or not cols:
        return A1Result(
            epsilon=epsilon, clean_decision=clean_decision, attacked_decision=clean_decision,
            clean_evidence_total=clean_evidence, attacked_evidence_total=clean_evidence,
            evaded=(clean_decision == benign_class_id), steps_run=0,
        )

    x_cur = x0.copy()
    lr = epsilon * lr_frac

    for _ in range(steps):
        delta = rng.choice([-1.0, 1.0], size=len(cols))
        base = x_cur[cols].to_numpy(dtype=float)

        plus = x_cur.copy()
        plus.loc[:, cols] = np.clip(base + spsa_c * np.abs(base) * delta, lo_bound, hi_bound)
        plus = project(plus)

        minus = x_cur.copy()
        minus.loc[:, cols] = np.clip(base - spsa_c * np.abs(base) * delta, lo_bound, hi_bound)
        minus = project(minus)

        _, _, p_plus = evaluate_raw_row(
            model, batch, plus, pipeline, feature_names, target_index_within_bin, device
        )
        _, _, p_minus = evaluate_raw_row(
            model, batch, minus, pipeline, feature_names, target_index_within_bin, device
        )
        loss_plus = _benign_neg_log_prob(p_plus, benign_class_id)
        loss_minus = _benign_neg_log_prob(p_minus, benign_class_id)

        denom = 2.0 * spsa_c * np.abs(base)
        denom[denom == 0.0] = 1.0
        ghat = (loss_plus - loss_minus) / denom * delta  # elementwise SPSA estimate

        step = base - lr * np.abs(base) * np.sign(ghat)  # descend the benign-neg-log-loss
        x_cur.loc[:, cols] = np.clip(step, lo_bound, hi_bound)
        x_cur = project(x_cur)

    attacked_decision, attacked_evidence, _ = evaluate_raw_row(
        model, batch, x_cur, pipeline, feature_names, target_index_within_bin, device
    )
    return A1Result(
        epsilon=epsilon,
        clean_decision=clean_decision,
        attacked_decision=attacked_decision,
        clean_evidence_total=clean_evidence,
        attacked_evidence_total=attacked_evidence,
        evaded=(attacked_decision == benign_class_id),
        steps_run=steps,
    )


def run_a1_epsilon_sweep(
    model: ArgusModel,
    pipeline,
    batch: dict,
    raw_row: pd.DataFrame,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
    epsilons: list[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    steps: int = 40,
    benign_class_id: int = 0,
    seed: int = 0,
) -> list[A1Result]:
    return [
        run_a1_pgd(
            model, pipeline, batch, raw_row, feature_names, target_index_within_bin,
            device, epsilon=eps, steps=steps, benign_class_id=benign_class_id, seed=seed,
        )
        for eps in epsilons
    ]
