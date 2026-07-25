"""Explanation-quality metrics: fidelity+/-, sparsity, necessity, stability.

See docs/11_XAI.md §4. All are computed against the actual model/batch (not
simulated), on the target flow's own feature vector (`target_edge_attr`) —
the same substitution mechanism used by `attacks/raw_evasion_common.py` and
`xai/explainers.py`'s edge-masking, for consistency across the codebase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch


def _predict_prob(
    model: ArgusModel, inputs: list, target_edge_attr_row: torch.Tensor, target_idx: int, class_idx: int
) -> float:
    local_inputs = list(inputs)
    tea = inputs[-1].clone()
    tea[target_idx] = target_edge_attr_row
    local_inputs[-1] = tea
    model.eval()
    with torch.no_grad():
        outputs = model(*local_inputs)
    return float(outputs["p_hat"][target_idx, class_idx].item())


def _topk_mask(attribution: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask, True for the top-k |attribution| entries."""
    k = min(k, len(attribution))
    order = np.argsort(-np.abs(attribution))
    mask = np.zeros(len(attribution), dtype=bool)
    mask[order[:k]] = True
    return mask


@dataclass
class FidelityResult:
    fidelity_plus: float
    fidelity_minus: float
    k: int
    sparsity: float


def feature_fidelity(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    class_idx: int,
    attribution: np.ndarray,
    baseline: torch.Tensor,
    k: int = 15,
) -> FidelityResult:
    """Fidelity+/- for a feature-level attribution over `target_edge_attr`.

    fidelity+ : drop in p(class) when the top-k attributed features are
        REMOVED (replaced by `baseline`). Large = explanation found what mattered.
    fidelity- : drop in p(class) when everything EXCEPT the top-k is removed.
        Small = the top-k alone is sufficient to reproduce the decision.
    """
    inputs = list(model_inputs_from_batch(batch, device))
    x = inputs[-1][target_idx].clone()
    baseline = baseline.to(x.device, dtype=x.dtype)
    mask = _topk_mask(attribution, k)
    mask_t = torch.as_tensor(mask, device=x.device)

    p_clean = _predict_prob(model, inputs, x, target_idx, class_idx)

    x_removed_topk = torch.where(mask_t, baseline, x)
    p_removed = _predict_prob(model, inputs, x_removed_topk, target_idx, class_idx)

    x_keep_topk_only = torch.where(mask_t, x, baseline)
    p_kept = _predict_prob(model, inputs, x_keep_topk_only, target_idx, class_idx)

    return FidelityResult(
        fidelity_plus=p_clean - p_removed,
        fidelity_minus=p_clean - p_kept,
        k=min(k, len(attribution)),
        sparsity=1.0 - min(k, len(attribution)) / len(attribution),
    )


def necessity_rate(flips: list[bool]) -> float:
    """Fraction of counterfactual explanations whose removal flipped the
    prediction (docs §4, "Necessity (ProvX)")."""
    if not flips:
        return 0.0
    return float(np.mean(flips))


def stability(attributions: list[np.ndarray]) -> float:
    """Mean pairwise Spearman rank correlation across attributions computed
    for small input perturbations that do not change the prediction (docs §4,
    "the metric that separates useful from decorative explanations").

    Args:
        attributions: list of attribution vectors (same features, same order),
            one per perturbation, including the un-perturbed one.
    Returns:
        Mean pairwise Spearman correlation in [-1, 1]; higher is more stable.
    """
    n = len(attributions)
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(attributions[i], attributions[j])
            if np.isfinite(rho):
                corrs.append(rho)
    return float(np.mean(corrs)) if corrs else 0.0


def timed_runtime_ms(fn, *args, **kwargs) -> tuple[float, object]:
    """Run `fn(*args, **kwargs)` once, returning (elapsed_ms, result)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result
