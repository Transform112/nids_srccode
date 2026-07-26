"""Native evidence attribution: the C4 evidence base.

See docs/11_XAI.md §1. Three levels:
1. Embedding-level (exact, no approximation): log e_k decomposes coordinate-wise
   over the embedding dimensions.
2. Feature-level (Integrated Gradients): propagate embedding attribution back
   to the target edge's own input features, along a path from a baseline.
3. Neighbour/edge-level (attr_edge, §1.2): attention weight x gradient
   sensitivity per neighbour message, normalised to sum 1 — the metric behind
   figure F7's injected_mass_fraction (C4 <-> C2 linkage).
"""

from __future__ import annotations

import torch


def embedding_attribution(z: torch.Tensor, prototype: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    """Exact per-embedding-dimension contribution to log e_k.

    log e_k = (1/tau) * <z, p_k> - (1-m)/tau  (constant w.r.t. z)
    so d(log e_k)/dz[j] contribution is z[j] * p_k[j] / tau (exact, no approximation).

    Args:
        z: [..., d_z] unit-norm embedding(s)
        prototype: [d_z] unit-norm prototype (or [..., d_z] broadcastable)
        tau: evidential temperature (scalar or tensor)
    Returns:
        [..., d_z] per-dimension attribution; sums to <z, p_k>/tau along the
        last axis (the part of log e_k that varies with z).
    """
    tau = tau if isinstance(tau, torch.Tensor) else torch.tensor(tau)
    return z * prototype / tau.clamp_min(1e-6)


def verify_embedding_decomposition(
    z: torch.Tensor, prototype: torch.Tensor, tau: torch.Tensor | float, margin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attributed_log_e, actual_log_e) for a completeness check.

    log e_k = (<z,p_k> - 1 + m) / tau   (since d_k = 1 - <z,p_k>)
    attributed_log_e = sum_j attribution[j] + (m - 1)/tau  should equal actual_log_e.
    """
    tau_t = tau if isinstance(tau, torch.Tensor) else torch.tensor(tau)
    attribution = embedding_attribution(z, prototype, tau_t)
    const = (margin - 1.0) / tau_t.clamp_min(1e-6)
    attributed_log_e = attribution.sum(dim=-1) + const
    cos = (z * prototype).sum(dim=-1)
    d = 1.0 - cos
    actual_log_e = -(d - margin) / tau_t.clamp_min(1e-6)
    return attributed_log_e, actual_log_e


def integrated_gradients_feature_attribution(
    forward_log_e_fn,
    target_edge_attr: torch.Tensor,
    baseline: torch.Tensor,
    class_idx: int,
    steps: int = 50,
) -> tuple[torch.Tensor, float, float]:
    """Integrated Gradients attribution of log e_{class_idx} to input features.

    Args:
        forward_log_e_fn: callable(edge_attr: Tensor[1, F_e]) -> Tensor[1, C]
            log-evidence, holding all other model inputs (graph context) fixed.
        target_edge_attr: [F_e] the actual input.
        baseline: [F_e] the baseline input (e.g. median benign flow).
        class_idx: which class's log-evidence to attribute.
        steps: number of integration steps.
    Returns:
        (attribution [F_e], f(x) at class_idx, f(baseline) at class_idx) — the
        completeness check is `attribution.sum() ≈ f(x) - f(baseline)`.
    """
    diff = target_edge_attr - baseline
    total_grad = torch.zeros_like(target_edge_attr)
    for step in range(1, steps + 1):
        alpha = step / steps
        x_interp = (baseline + alpha * diff).clone().detach().requires_grad_(True)
        log_e = forward_log_e_fn(x_interp.unsqueeze(0))[0, class_idx]
        (grad,) = torch.autograd.grad(log_e, x_interp)
        total_grad += grad
    avg_grad = total_grad / steps
    attribution = diff * avg_grad

    with torch.no_grad():
        f_x = forward_log_e_fn(target_edge_attr.unsqueeze(0))[0, class_idx].item()
        f_base = forward_log_e_fn(baseline.unsqueeze(0))[0, class_idx].item()
    return attribution, f_x, f_base


def victim_slot_flags(
    edge_index: torch.Tensor, victim_local: int, k: int, edge_flags: torch.Tensor
) -> torch.Tensor:
    """Per-neighbour-slot flags for the victim node, in dense-slot order.

    Mirrors `srteg._edges_to_dense`'s slot assignment (first `k` incident
    edges, in edge order), so element j aligns with attention weight j and
    message-gradient row j for that node.
    """
    dst = edge_index[1]
    e_idx = (dst == victim_local).nonzero(as_tuple=True)[0][:k]
    return edge_flags[e_idx]


def edge_attribution_for_victim(
    model,
    batch: dict,
    victim_local: int,
    target_pos: int,
    device: torch.device,
    class_idx: int | None = None,
) -> torch.Tensor | None:
    """attr_edge over the victim's long-scale neighbourhood (docs/11_XAI.md §1.2).

    attr_edge[n] = alpha_{n,v} * || d(score)/d m_{n->v} ||_2, normalised to sum
    to 1. `score` is the target flow's predicted-class logit (log p-hat) — the
    doc's full Jacobian norm ||d h_e / d m|| is replaced by the gradient of
    this scalar decision score, the standard saliency reduction (disclosed;
    the Jacobian needs d_h backward passes per neighbour).

    Attention weights and message gradients come from the LAST GNN layer of
    the LONG scale — the window A2 injects into and the last forward_scale
    call, so the layer side-channels hold exactly that scale's tensors.

    Returns [n_v] attribution over the victim's kept neighbour slots, or None
    if the victim has no long-scale neighbours.
    """
    from argus.train.loop import model_inputs_from_batch

    last_layer = model.encoder.gnn_layers[-1]
    was_training = model.training
    model.eval()
    last_layer.record_attention = True
    last_layer.record_messages = True
    try:
        inputs = model_inputs_from_batch(batch, device)
        with torch.enable_grad():
            outputs = model(*inputs)
            if class_idx is None:
                class_idx = int(outputs["logits"][target_pos].argmax().item())
            score = outputs["logits"][target_pos, class_idx]
            model.zero_grad(set_to_none=True)
            score.backward()

        alpha = last_layer.last_attn.get(victim_local)
        msgs = last_layer.last_msgs
        if alpha is None or msgs is None or msgs.grad is None:
            return None
        n_v = alpha.shape[0]
        grad_norms = msgs.grad[victim_local, :n_v].norm(dim=-1)  # [n_v]
        attr = alpha * grad_norms
        total = attr.sum()
        if total <= 0:
            return torch.full_like(attr, 1.0 / max(n_v, 1))
        return attr / total
    finally:
        last_layer.record_attention = False
        last_layer.record_messages = False
        last_layer.last_msgs = None
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def injected_mass_fraction(attr_edge: torch.Tensor, injected_slots: torch.Tensor) -> float:
    """Fraction of (already-normalised) attribution mass on injected edges (F7)."""
    if attr_edge.numel() == 0:
        return 0.0
    return float(attr_edge[injected_slots.to(attr_edge.device)].sum().item())
