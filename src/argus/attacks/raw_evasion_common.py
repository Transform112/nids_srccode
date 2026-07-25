"""Shared machinery for attacks that perturb a *single target flow's own raw
features* (A1 feature evasion, A5 temporal jitter) rather than injecting new
context flows (A2).

Both attacks reuse the frozen `FeaturePipeline` to recompute the full,
internally-consistent transformed feature vector (including TE2 derived
features) after every raw-space perturbation, then substitute only the
target flow's row of `target_edge_attr` before a forward pass. The rest of
the bin batch (context edges, node features) is built once and reused,
since only the target flow's own attributes change — the neighbourhood
context is untouched by these two attacks (that is A2's job).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch


def evaluate_raw_row_full(
    model: ArgusModel,
    batch: dict,
    raw_row: pd.DataFrame,
    pipeline,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
) -> tuple[int, dict, int]:
    """As `evaluate_raw_row`, but returns the full head-output dict (so callers
    that need `cos_c`, `evidence_total`, `z`, etc. — e.g. A4's adaptive
    attacker — don't need a separate forward pass).

    Returns:
        (decision, outputs, idx) — `idx` is the target's row within `outputs`.
    """
    transformed = pipeline.transform(raw_row)
    vec = transformed[feature_names].to_numpy(dtype=np.float32)[0]

    inputs = list(model_inputs_from_batch(batch, device))
    target_edge_attr = inputs[-1].clone()
    idx = min(target_index_within_bin, target_edge_attr.shape[0] - 1)
    target_edge_attr[idx] = torch.as_tensor(vec, device=device, dtype=target_edge_attr.dtype)
    inputs[-1] = target_edge_attr

    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)

    if hasattr(model.head, "decide"):
        decisions, _ = model.head.decide(outputs)
        decision = int(decisions[idx].item())
    else:
        decision = int(outputs["p_hat"][idx].argmax().item())
    return decision, outputs, idx


def evaluate_raw_row(
    model: ArgusModel,
    batch: dict,
    raw_row: pd.DataFrame,
    pipeline,
    feature_names: list[str],
    target_index_within_bin: int,
    device: torch.device,
) -> tuple[int, float, torch.Tensor]:
    """Transform `raw_row` through the frozen pipeline, substitute it into the
    target flow's slot of `batch["target_edge_attr"]`, and run the model.

    Returns:
        (decision, evidence_total, p_hat_row)
    """
    decision, outputs, idx = evaluate_raw_row_full(
        model, batch, raw_row, pipeline, feature_names, target_index_within_bin, device
    )
    evidence_total = float(
        outputs.get("evidence_total", outputs["p_hat"].max(dim=1).values)[idx].item()
    )
    return decision, evidence_total, outputs["p_hat"][idx].detach().cpu()
