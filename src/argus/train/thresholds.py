"""Validation-only threshold selection for theta_unknown / theta_defer.

See docs/05_ARCHITECTURE.md §6.4 and docs/07_HYPERPARAMETERS.md.
Thresholds are selected on validation only; test is touched once, at the end.
"""

from __future__ import annotations

import numpy as np
import torch

from argus.models.epc import EPCHead


def select_theta_unknown(
    evidence_totals: torch.Tensor,
    target_false_unknown_rate: float = 0.05,
) -> float:
    """Select theta_unknown for a target false-UNKNOWN rate on known-class validation data.

    `evidence_totals` are E_total for known-class validation flows only. The
    threshold is set so that `target_false_unknown_rate` fraction of them would
    be (incorrectly) flagged UNKNOWN.
    """
    values = evidence_totals.detach().cpu().numpy()
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, target_false_unknown_rate))


def select_theta_defer(
    margins: torch.Tensor,
    target_defer_rate: float = 0.02,
) -> float:
    """Select theta_defer for a target deferral rate on validation.

    `margins` are p_max - p_2nd for known-class validation flows that were not
    flagged UNKNOWN. The threshold is set so `target_defer_rate` fraction of
    them would be deferred.
    """
    values = margins.detach().cpu().numpy()
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, target_defer_rate))


def collect_validation_evidence(
    model,
    head: EPCHead,
    val_source,
    device: torch.device,
    max_bins: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over validation and collect evidence totals + margins.

    Returns:
        (evidence_totals, margins) each [N] over all validation targets.
    """
    from argus.train.loop import model_inputs_from_batch

    model.eval()
    all_evidence, all_margins = [], []
    bins = val_source.unique_bins[:max_bins] if max_bins else val_source.unique_bins
    with torch.no_grad():
        for bin_id in bins:
            batch = val_source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            inputs = model_inputs_from_batch(batch, device)
            outputs = model(*inputs)
            all_evidence.append(outputs["evidence_total"].cpu())
            top2 = torch.topk(outputs["p_hat"], k=min(2, outputs["p_hat"].shape[1]), dim=1).values
            if top2.shape[1] == 2:
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = top2[:, 0]
            all_margins.append(margin.cpu())
    if not all_evidence:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(all_evidence), torch.cat(all_margins)


def calibrate_thresholds(
    model,
    val_source,
    device: torch.device,
    target_false_unknown_rate: float = 0.05,
    target_defer_rate: float = 0.02,
    max_bins: int | None = None,
) -> dict[str, float]:
    """Full threshold-calibration routine: run validation, select both thresholds,
    and write them onto the head so `decide()` uses them by default.
    """
    head = model.head
    evidence_totals, margins = collect_validation_evidence(model, head, val_source, device, max_bins)
    theta_unknown = select_theta_unknown(evidence_totals, target_false_unknown_rate)
    theta_defer = select_theta_defer(margins, target_defer_rate)
    if hasattr(head, "theta_unknown"):
        head.theta_unknown = theta_unknown
    if hasattr(head, "theta_defer"):
        head.theta_defer = theta_defer
    return {"theta_unknown": theta_unknown, "theta_defer": theta_defer}
