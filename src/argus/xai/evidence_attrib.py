"""Native evidence attribution: the C4 evidence base.

See docs/11_XAI.md §1. Two levels:
1. Embedding-level (exact, no approximation): log e_k decomposes coordinate-wise
   over the embedding dimensions.
2. Feature-level (Integrated Gradients): propagate embedding attribution back
   to the target edge's own input features, along a path from a baseline.
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
