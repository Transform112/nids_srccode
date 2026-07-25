"""Baseline explainers, for comparison against native evidence attribution.

See docs/11_XAI.md §2. All operate on the *inputs* to a fixed, frozen model
(mask/perturb `edge_attr` / `target_edge_attr` before a forward pass) rather
than requiring any change to model internals — a standard, correct formulation
for perturbation-based explainers (GNNExplainer, KernelSHAP) and a safe way to
implement PGExplainer without invasive changes to `SRTEGEncoder`.

The one exception is `attention_weights_baseline`, which reads the model's own
attention weights via the opt-in recording side-channel added to
`SRTEGLayer` (`record_attention`/`last_attn`) — this is a naive, non-learned
"baseline" precisely because it is not an explanation method at all; docs §2
notes it is included specifically to show *where it disagrees* with the
learned/derived explainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch

_SCALE_EDGE_ATTR_POS = {"short": 2, "mid": 5, "long": 8}  # positions of edge_attr_* in model_inputs_from_batch
_SCALE_EDGE_INDEX_POS = {"short": 1, "mid": 4, "long": 7}


def attention_weights_baseline(
    model: ArgusModel, batch: dict, device: torch.device, layer_idx: int = -1
) -> dict[int, torch.Tensor]:
    """Return the model's own last-layer time-decayed attention weights per
    node (naive "explanation" baseline — attention is not attribution).

    Note: since GNN layers are weight-shared across scales, this captures
    whichever scale is processed *last* in `ArgusModel.forward` (the long
    scale) for the chosen layer.
    """
    layers = list(model.encoder.gnn_layers)
    layer = layers[layer_idx]
    layer.record_attention = True
    try:
        inputs = model_inputs_from_batch(batch, device)
        model.eval()
        with torch.no_grad():
            model(*inputs)
        return dict(layer.last_attn)
    finally:
        layer.record_attention = False
        layer.last_attn = {}


def _forward_with_masked_edges(
    model: ArgusModel, inputs: list, masks: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Re-run the model with each scale's `edge_attr` scaled by `masks[scale]`
    (a soft edge-drop relaxation: a fully-masked edge contributes nothing to
    message construction, which is the standard continuous relaxation
    GNNExplainer-style methods optimise over).
    """
    masked_inputs = list(inputs)
    for scale, pos in _SCALE_EDGE_ATTR_POS.items():
        if scale in masks:
            masked_inputs[pos] = inputs[pos] * masks[scale].unsqueeze(-1)
    return model(*masked_inputs)


@dataclass
class EdgeMaskResult:
    masks: dict[str, torch.Tensor]
    final_loss: float
    epochs_run: int


def _mask_learning_loop(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    class_idx: int,
    mask_fn,
    params: list[nn.Parameter],
    epochs: int,
    lr: float,
    lam_sparsity: float,
    lam_entropy: float,
) -> EdgeMaskResult:
    inputs = list(model_inputs_from_batch(batch, device))
    opt = torch.optim.Adam(params, lr=lr)
    model.eval()  # frozen model; only the mask (or mask-generating MLP) is optimised
    for p in model.parameters():
        p.requires_grad_(False)

    final_loss = 0.0
    for epoch in range(epochs):
        opt.zero_grad()
        masks = mask_fn()
        outputs = _forward_with_masked_edges(model, inputs, masks)
        log_p = torch.log(outputs["p_hat"][target_idx, class_idx].clamp_min(1e-12))
        sparsity = sum(m.mean() for m in masks.values()) / max(len(masks), 1)
        entropy = sum(
            -(m.clamp(1e-6, 1 - 1e-6) * torch.log(m.clamp(1e-6, 1 - 1e-6))
              + (1 - m).clamp(1e-6, 1 - 1e-6) * torch.log((1 - m).clamp(1e-6, 1 - 1e-6))).mean()
            for m in masks.values()
        ) / max(len(masks), 1)
        loss = -log_p + lam_sparsity * sparsity + lam_entropy * entropy
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        final_masks = {k: v.detach().clamp(0, 1) for k, v in mask_fn().items()}
    return EdgeMaskResult(masks=final_masks, final_loss=final_loss, epochs_run=epochs)


def gnnexplainer_edge_mask(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    class_idx: int,
    epochs: int = 200,
    lr: float = 0.05,
    lam_sparsity: float = 0.01,
    lam_entropy: float = 0.01,
    scales: tuple[str, ...] = ("short", "mid", "long"),
) -> EdgeMaskResult:
    """GNNExplainer-style mask learning: a free per-edge mask parameter per
    scale, optimised to preserve the target's predicted-class probability
    while being as sparse and as close to binary as possible (docs §2, 200
    epochs default).
    """
    inputs = model_inputs_from_batch(batch, device)
    raw_masks: dict[str, nn.Parameter] = {}
    for scale in scales:
        n_edges = inputs[_SCALE_EDGE_ATTR_POS[scale]].shape[0]
        raw_masks[scale] = nn.Parameter(torch.zeros(n_edges, device=device) + 2.0)  # sigmoid(2) ~ 0.88, start open

    def mask_fn():
        return {scale: torch.sigmoid(raw_masks[scale]) for scale in scales}

    return _mask_learning_loop(
        model, batch, device, target_idx, class_idx, mask_fn,
        params=list(raw_masks.values()), epochs=epochs, lr=lr,
        lam_sparsity=lam_sparsity, lam_entropy=lam_entropy,
    )


class _MaskMLP(nn.Module):
    """Tiny per-edge mask-logit predictor: mask = sigmoid(MLP(edge_attr)).

    This is the defining difference from GNNExplainer: the mask is a function
    of edge features rather than a free parameter per edge, matching
    PGExplainer's core idea. A single small MLP is trained per-instance here
    (a simplified stand-in for PGExplainer's cross-instance amortised
    training — see module note; full amortisation needs many training
    instances and is out of scope for a per-decision explainer call).
    """

    def __init__(self, f_e: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f_e, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(edge_attr).squeeze(-1))


def pgexplainer_edge_mask(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    class_idx: int,
    f_e: int,
    epochs: int = 30,
    lr: float = 0.05,
    lam_sparsity: float = 0.01,
    lam_entropy: float = 0.01,
    scales: tuple[str, ...] = ("short", "mid", "long"),
) -> EdgeMaskResult:
    """Simplified PGExplainer: masks are predicted by a small MLP over each
    edge's own features rather than being free parameters (docs §2, 30 epochs
    default — faster than GNNExplainer's 200 because the search space is the
    MLP's ~300 parameters, not one parameter per edge).
    """
    inputs = model_inputs_from_batch(batch, device)
    mlps = {scale: _MaskMLP(f_e).to(device) for scale in scales}

    def mask_fn():
        return {scale: mlps[scale](inputs[_SCALE_EDGE_ATTR_POS[scale]]) for scale in scales}

    params = [p for mlp in mlps.values() for p in mlp.parameters()]
    return _mask_learning_loop(
        model, batch, device, target_idx, class_idx, mask_fn,
        params=params, epochs=epochs, lr=lr,
        lam_sparsity=lam_sparsity, lam_entropy=lam_entropy,
    )


def kernelshap_feature_importance(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    class_idx: int,
    baseline: torch.Tensor,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """KernelSHAP over the target flow's own input features only (docs §2:
    "features only", 200 samples default). Coalition sampling + the standard
    SHAP kernel weight, solved by weighted least squares.
    """
    rng = np.random.default_rng(seed)
    f_e = baseline.shape[0]
    inputs = list(model_inputs_from_batch(batch, device))
    target_edge_attr = inputs[-1].clone()
    x = target_edge_attr[target_idx].clone()

    def predict(mask_row: np.ndarray) -> float:
        mask_t = torch.as_tensor(mask_row, device=device, dtype=x.dtype)
        candidate = mask_t * x + (1 - mask_t) * baseline.to(device)
        local_inputs = list(inputs)
        tea = target_edge_attr.clone()
        tea[target_idx] = candidate
        local_inputs[-1] = tea
        with torch.no_grad():
            outputs = model(*local_inputs)
        return float(outputs["p_hat"][target_idx, class_idx].item())

    masks = rng.integers(0, 2, size=(n_samples, f_e)).astype(np.float64)
    masks[0] = 1.0  # full coalition
    masks[1] = 0.0  # empty coalition
    y = np.array([predict(m) for m in masks])

    sizes = masks.sum(axis=1)
    # SHAP kernel weight; the empty/full coalitions get a large fixed weight
    # (standard KernelSHAP practice, since the kernel is undefined there).
    weights = np.full(n_samples, 1e6)
    interior = (sizes > 0) & (sizes < f_e)
    for i in np.nonzero(interior)[0]:
        s = int(sizes[i])
        weights[i] = (f_e - 1) / (s * (f_e - s))

    design = np.concatenate([np.ones((n_samples, 1)), masks], axis=1)
    wsqrt = np.sqrt(weights)
    design_w = design * wsqrt[:, None]
    y_w = y * wsqrt
    coef, *_ = np.linalg.lstsq(design_w, y_w, rcond=None)
    return coef[1:]  # drop the intercept; per-feature Shapley estimates


@dataclass
class CounterfactualResult:
    flipped: bool
    n_removed: int
    removed_fraction: float
    removed_scale_edge_indices: dict[str, list[int]] = field(default_factory=dict)


def counterfactual_necessity(
    model: ArgusModel,
    batch: dict,
    device: torch.device,
    target_idx: int,
    ranked_edges: dict[str, np.ndarray],
    max_removed_fraction: float = 1.0,
) -> CounterfactualResult:
    """Greedily remove context edges in the order given by `ranked_edges`
    (highest-attribution first, per scale) until the prediction flips away
    from its clean argmax class, or the budget is exhausted (docs §2,
    "minimal edge subset whose removal flips the prediction").

    Args:
        ranked_edges: {scale: array of edge positions within that scale's
            edge_attr, ordered most-to-least important}.
    """
    inputs = list(model_inputs_from_batch(batch, device))
    with torch.no_grad():
        clean_outputs = model(*inputs)
    clean_class = int(clean_outputs["p_hat"][target_idx].argmax().item())

    total_edges = sum(len(v) for v in ranked_edges.values())
    if total_edges == 0:
        return CounterfactualResult(flipped=False, n_removed=0, removed_fraction=0.0)

    removed: dict[str, list[int]] = {scale: [] for scale in ranked_edges}
    cursors = {scale: 0 for scale in ranked_edges}
    max_removed = int(max_removed_fraction * total_edges)

    for step in range(1, max_removed + 1):
        # Remove the single next-most-important edge across all scales (by
        # rank position, interleaved round-robin so no scale dominates).
        scale = list(ranked_edges.keys())[(step - 1) % len(ranked_edges)]
        if cursors[scale] < len(ranked_edges[scale]):
            removed[scale].append(int(ranked_edges[scale][cursors[scale]]))
            cursors[scale] += 1

        masked_inputs = list(inputs)
        for scale, idxs in removed.items():
            if not idxs:
                continue
            pos = _SCALE_EDGE_ATTR_POS[scale]
            mask = torch.ones(inputs[pos].shape[0], device=device)
            mask[torch.as_tensor(idxs, device=device, dtype=torch.long)] = 0.0
            masked_inputs[pos] = inputs[pos] * mask.unsqueeze(-1)

        with torch.no_grad():
            outputs = model(*masked_inputs)
        new_class = int(outputs["p_hat"][target_idx].argmax().item())
        n_removed_total = sum(len(v) for v in removed.values())
        if new_class != clean_class:
            return CounterfactualResult(
                flipped=True, n_removed=n_removed_total,
                removed_fraction=n_removed_total / total_edges,
                removed_scale_edge_indices=removed,
            )

    n_removed_total = sum(len(v) for v in removed.values())
    return CounterfactualResult(
        flipped=False, n_removed=n_removed_total,
        removed_fraction=n_removed_total / total_edges,
        removed_scale_edge_indices=removed,
    )
