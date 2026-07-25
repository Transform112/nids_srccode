"""A4 — adaptive white-box attacker.

See docs/10_ADVERSARIAL.md §5. Full knowledge of weights, prototype bank,
thresholds, and the three-way decision rule; optimises the embedding directly
toward the benign prototype. Restricted to Channel A (controllable) raw
columns **plus timing** (IAT), since A4 is explicitly allowed to pay the A5
timing cost too (docs §1.2, §5).

Same SPSA-through-the-frozen-pipeline mechanics as A1 (see
`a1_feature_pgd.py` module docstring for why: the pipeline's quantile
transform and one-hot encoders are not autograd-traceable, so the ascent
direction is estimated rather than backpropagated). "White-box" here means
full access to the prototype bank, thresholds and decision rule when defining
the *objective* — not that literal analytic gradients are used.

Two objectives:
    - `evasion`: minimise distance to the benign prototype, `1 - cos(z, p_benign)`.
    - `unknown_avoidance`: additionally maximise total evidence, so a
      successful evasion lands on a confident benign classification rather
      than UNKNOWN (an UNKNOWN verdict still alerts — see docs §1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from argus.attacks.a1_feature_pgd import RAW_NUMERIC_CONTROLLABLE
from argus.attacks.a5_temporal_jitter import IAT_PREFIXES, IAT_SUFFIXES
from argus.attacks.constraints import project
from argus.attacks.raw_evasion_common import evaluate_raw_row_full
from argus.models.argus import ArgusModel

RAW_A4_COLUMNS = list(RAW_NUMERIC_CONTROLLABLE) + [
    f"{prefix}{suffix}" for prefix in IAT_PREFIXES for suffix in IAT_SUFFIXES
]


@dataclass
class A4Result:
    objective: str
    clean_decision: int
    attacked_decision: int
    clean_cos_benign: float
    attacked_cos_benign: float
    clean_evidence_total: float
    attacked_evidence_total: float
    evaded: bool
    steps_run: int


def _objective_loss(outputs: dict, idx: int, benign_class_id: int, objective: str, uev_weight: float) -> float:
    cos_benign = float(outputs["cos_c"][idx, benign_class_id].item())
    dist = 1.0 - cos_benign
    if objective == "evasion":
        return dist
    if objective == "unknown_avoidance":
        evidence_total = float(outputs["evidence_total"][idx].item())
        return dist - uev_weight * np.log1p(max(evidence_total, 0.0))
    raise ValueError(f"Unknown A4 objective: {objective}")


def run_a4_adaptive(
    model: ArgusModel,
    pipeline,
    batch: dict,
    raw_row: pd.DataFrame,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
    objective: str = "evasion",
    steps: int = 100,
    epsilon: float = 0.5,
    spsa_c: float = 0.1,
    lr_frac: float = 0.15,
    uev_weight: float = 0.1,
    benign_class_id: int = 0,
    seed: int = 0,
) -> A4Result:
    """Run the adaptive attacker on a single target flow.

    `epsilon` bounds each perturbed raw column to ±epsilon of its clean value
    (same relative-budget convention as A1), but the budget is generous by
    default (0.5) because A4 is meant to expose the worst case, not a
    realistic-cost scenario — see docs §5, "expect it to be the most
    successful attack".
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in RAW_A4_COLUMNS if c in raw_row.columns]

    x0 = raw_row.copy().reset_index(drop=True)
    if cols:
        x0[cols] = x0[cols].astype(np.float64)
    lo_bound = x0[cols].to_numpy(dtype=float) * (1.0 - epsilon)
    hi_bound = x0[cols].to_numpy(dtype=float) * (1.0 + epsilon)

    clean_decision, clean_outputs, idx = evaluate_raw_row_full(
        model, batch, x0, pipeline, feature_names, target_index_within_bin, device
    )
    clean_cos_benign = float(clean_outputs["cos_c"][idx, benign_class_id].item())
    clean_evidence = float(clean_outputs["evidence_total"][idx].item())

    if not cols:
        return A4Result(
            objective=objective, clean_decision=clean_decision, attacked_decision=clean_decision,
            clean_cos_benign=clean_cos_benign, attacked_cos_benign=clean_cos_benign,
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

        _, out_plus, idx_p = evaluate_raw_row_full(
            model, batch, plus, pipeline, feature_names, target_index_within_bin, device
        )
        _, out_minus, idx_m = evaluate_raw_row_full(
            model, batch, minus, pipeline, feature_names, target_index_within_bin, device
        )
        loss_plus = _objective_loss(out_plus, idx_p, benign_class_id, objective, uev_weight)
        loss_minus = _objective_loss(out_minus, idx_m, benign_class_id, objective, uev_weight)

        denom = 2.0 * spsa_c * np.abs(base)
        denom[denom == 0.0] = 1.0
        ghat = (loss_plus - loss_minus) / denom * delta

        step = base - lr * np.abs(base) * np.sign(ghat)
        x_cur.loc[:, cols] = np.clip(step, lo_bound, hi_bound)
        x_cur = project(x_cur)

    attacked_decision, attacked_outputs, idx_f = evaluate_raw_row_full(
        model, batch, x_cur, pipeline, feature_names, target_index_within_bin, device
    )
    return A4Result(
        objective=objective,
        clean_decision=clean_decision,
        attacked_decision=attacked_decision,
        clean_cos_benign=clean_cos_benign,
        attacked_cos_benign=float(attacked_outputs["cos_c"][idx_f, benign_class_id].item()),
        clean_evidence_total=clean_evidence,
        attacked_evidence_total=float(attacked_outputs["evidence_total"][idx_f].item()),
        evaded=(attacked_decision == benign_class_id),
        steps_run=steps,
    )
