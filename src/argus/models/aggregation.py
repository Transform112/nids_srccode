"""Robust aggregation operators and multi-aggregator readout.

See docs/05_ARCHITECTURE.md §3.5 and §3.5b.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_weights(msgs: torch.Tensor, weights: torch.Tensor) -> None:
    if msgs.ndim != 2:
        raise ValueError("msgs must be [n, d]")
    if weights.ndim != 1 or weights.shape[0] != msgs.shape[0]:
        raise ValueError("weights must be [n]")


def trimmed_mean(msgs: torch.Tensor, weights: torch.Tensor, beta: float = 0.20) -> torch.Tensor:
    """Coordinate-wise trimmed mean.

    Args:
        msgs: [n, d]
        weights: [n] summing to 1
        beta: trim fraction per tail.
    Returns:
        [d] trimmed mean.
    """
    _check_weights(msgs, weights)
    n = msgs.shape[0]
    k = int(beta * n)
    if n - 2 * k < 1:
        return (weights.unsqueeze(-1) * msgs).sum(0)
    order = msgs.argsort(dim=0)
    keep = order[k : n - k]
    m = torch.gather(msgs, 0, keep)
    w = torch.gather(weights.unsqueeze(-1).expand_as(msgs), 0, keep)
    return (w * m).sum(0) / (w.sum(0) + 1e-9)


def trimmed_std(msgs: torch.Tensor, weights: torch.Tensor, beta: float = 0.20) -> torch.Tensor:
    """Coordinate-wise trimmed standard deviation."""
    _check_weights(msgs, weights)
    n = msgs.shape[0]
    k = int(beta * n)
    if n - 2 * k < 1:
        mean = (weights.unsqueeze(-1) * msgs).sum(0)
        var = (weights.unsqueeze(-1) * (msgs - mean) ** 2).sum(0)
        return torch.sqrt(var + 1e-9)
    order = msgs.argsort(dim=0)
    keep = order[k : n - k]
    m = torch.gather(msgs, 0, keep)
    w = torch.gather(weights.unsqueeze(-1).expand_as(msgs), 0, keep)
    mean = (w * m).sum(0) / (w.sum(0) + 1e-9)
    var = (w * (m - mean) ** 2).sum(0) / (w.sum(0) + 1e-9)
    return torch.sqrt(var + 1e-9)


def soft_medoid(
    msgs: torch.Tensor, weights: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Differentiable soft medoid.

    Args:
        msgs: [n, d]
        weights: [n]
        temperature: softmax temperature (floored at 0.05).
    Returns:
        [d] aggregate.
    """
    _check_weights(msgs, weights)
    T = max(temperature, 0.05)
    dist = torch.cdist(msgs, msgs, p=2)  # [n, n]
    total_dist = dist.sum(dim=1)  # [n]
    coeffs = F.softmax(-total_dist / T, dim=0)  # [n]
    denom = (coeffs * weights).sum() + 1e-9
    return ((coeffs * weights / denom).unsqueeze(-1) * msgs).sum(0)


def mean_aggregate(msgs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Plain weighted mean (E-GraphSAGE baseline)."""
    _check_weights(msgs, weights)
    return (weights.unsqueeze(-1) * msgs).sum(0)


# ---------------------------------------------------------------------------
# Batched variants.
#
# The per-node functions above operate on one node's [n_v, d] neighbour set and
# were called from a Python `for v in range(N)` loop in SRTEGLayer.forward.
# That loop cost ~8.25 ms/node — with ~1.22 M node-iterations per CICIDS2018
# epoch it dominated training wall-clock (~3 h/epoch; see docs/BUGS.md #48).
#
# These operate on the dense [N, K, d] message tensor the layer already builds,
# with a [N, K] validity mask, computing every node at once. Semantics are
# identical to the loop versions, including the `int(beta * n_v)` truncation and
# the `n_v - 2k < 1` fallback — `tests/test_vectorised_layer.py` asserts
# equivalence against them.
# ---------------------------------------------------------------------------


def _trim_bounds(mask: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node valid count, trim count, and whether trimming applies.

    `k` is computed in float64 so it matches Python's `int(beta * n)` exactly
    (float32 would round e.g. 0.2*15 across the truncation boundary).
    """
    n_v = mask.sum(dim=1)                                    # [N]
    k_v = (beta * n_v.double()).long()                       # [N] truncation
    use_trim = (n_v - 2 * k_v) >= 1                          # [N]
    return n_v, k_v, use_trim


def _sorted_keep(
    msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate-wise sort with masked entries pushed to the end, plus the
    keep-mask selecting ranks [k, n_v - k) — i.e. the trimmed middle."""
    n, k_dim, d = msgs.shape
    n_v, k_v, use_trim = _trim_bounds(mask, beta)

    big = torch.finfo(msgs.dtype).max
    msgs_pad = msgs.masked_fill(~mask.unsqueeze(-1), big)
    order = msgs_pad.argsort(dim=1)
    m_sorted = torch.gather(msgs_pad, 1, order)
    w_sorted = torch.gather(weights.unsqueeze(-1).expand(n, k_dim, d), 1, order)

    rank = torch.arange(k_dim, device=msgs.device).view(1, k_dim, 1)
    keep = (rank >= k_v.view(-1, 1, 1)) & (rank < (n_v - k_v).view(-1, 1, 1))
    keep = keep & use_trim.view(-1, 1, 1)

    zero = torch.zeros((), dtype=msgs.dtype, device=msgs.device)
    # `where` (not multiply) so the sentinel never participates in arithmetic.
    return torch.where(keep, m_sorted, zero), torch.where(keep, w_sorted, zero), use_trim


def trimmed_mean_std_batched(
    msgs: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.20,
    want_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batched trimmed mean and (optionally) std, **sharing one sort**.

    The sorted/gathered [N,K,d] tensors are the dominant memory cost of the
    batched path and autograd retains them until backward. Computing mean and
    std separately sorted twice and OOM'd a 16 GB T4 on chunks of large bins
    (docs/BUGS.md #48); sharing halves it. Returns `(mean, std)`, with `std`
    None when `want_std` is False.
    """
    m_k, w_k, use_trim = _sorted_keep(msgs, weights, mask, beta)
    keep_sum = w_k.sum(dim=1) + 1e-9
    trimmed_num = (w_k * m_k).sum(dim=1)
    trimmed_mean_ = trimmed_num / keep_sum

    wm = (weights.unsqueeze(-1) * msgs).masked_fill(~mask.unsqueeze(-1), 0.0)
    plain_mean = wm.sum(dim=1)
    use = use_trim.unsqueeze(-1)
    mean_out = torch.where(use, trimmed_mean_, plain_mean)

    if not want_std:
        return mean_out, None

    # The loop version's two branches differ: the fallback uses *unnormalised*
    # weighted sums, the trimmed path normalises. Preserved exactly.
    var_t = (w_k * (m_k - trimmed_mean_.unsqueeze(1)) ** 2).sum(dim=1) / keep_sum
    var_p = (
        (weights.unsqueeze(-1) * (msgs - plain_mean.unsqueeze(1)) ** 2)
        .masked_fill(~mask.unsqueeze(-1), 0.0)
        .sum(dim=1)
    )
    std_out = torch.sqrt(torch.where(use, var_t, var_p) + 1e-9)
    return mean_out, std_out


def trimmed_mean_batched(
    msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, beta: float = 0.20
) -> torch.Tensor:
    """Batched `trimmed_mean`. msgs [N,K,d], weights [N,K], mask [N,K] -> [N,d]."""
    return trimmed_mean_std_batched(msgs, weights, mask, beta, want_std=False)[0]


def trimmed_std_batched(
    msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, beta: float = 0.20
) -> torch.Tensor:
    """Batched `trimmed_std`."""
    return trimmed_mean_std_batched(msgs, weights, mask, beta, want_std=True)[1]


def mean_aggregate_batched(
    msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Batched `mean_aggregate`."""
    return (weights.unsqueeze(-1) * msgs).masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)


def soft_medoid_batched(
    msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Batched `soft_medoid`. Pairwise distances are [N,K,K]."""
    t = max(temperature, 0.05)
    pair_valid = mask.unsqueeze(1) & mask.unsqueeze(2)               # [N,K,K]
    dist = torch.cdist(msgs, msgs, p=2).masked_fill(~pair_valid, 0.0)
    total = dist.sum(dim=2)                                          # [N,K]
    coeffs = F.softmax(total.masked_fill(~mask, float("inf")).neg() / t, dim=1)
    coeffs = coeffs.masked_fill(~mask, 0.0)
    denom = (coeffs * weights).sum(dim=1, keepdim=True) + 1e-9
    return ((coeffs * weights / denom).unsqueeze(-1) * msgs).sum(dim=1)


class MultiAggregator(nn.Module):
    """Concatenates a robust aggregate, its spread, and a degree-scaled copy.

    Projects back to d_h. See docs/05_ARCHITECTURE.md §3.5b:
    ``agg_v = W_agg . concat[a_robust, a_spread, a_scale * a_robust]``.
    """

    def __init__(self, d_h: int, beta: float = 0.20, k: int = 32) -> None:
        super().__init__()
        self.d_h = d_h
        self.beta = beta
        self.k = k
        self.project = nn.Linear(3 * d_h, d_h, bias=True)

    def forward(self, msgs: torch.Tensor, weights: torch.Tensor, a_robust: torch.Tensor) -> torch.Tensor:
        """Args:
            msgs: [n, d_h]
            weights: [n]
            a_robust: [d_h] the aggregate `RobustAggregator` already computed
                (trimmed_mean or soft_medoid, whichever was configured) — the
                multi-aggregator readout must build on top of it, not silently
                substitute its own trimmed_mean regardless of configuration.
        Returns:
            [d_h]
        """
        a_spread = trimmed_std(msgs, weights, self.beta)
        n = msgs.shape[0]
        a_scale = math.log1p(n) / math.log1p(self.k)
        combined = torch.cat([a_robust, a_spread, a_scale * a_robust], dim=0)
        return self.project(combined)

    def forward_batched(
        self,
        msgs: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        a_robust: torch.Tensor,
        spread: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched `forward`. msgs [N,K,d_h], a_robust [N,d_h] -> [N,d_h].

        `a_scale` uses the per-node valid neighbour count, matching the loop
        version's `msgs.shape[0]` (which was that node's n_v, not K).
        `spread` may be supplied by the caller when it already computed the
        trimmed std from a shared sort, avoiding a second [N,K,d] sort.
        """
        a_spread = spread if spread is not None else trimmed_std_batched(
            msgs, weights, mask, self.beta
        )
        n_v = mask.sum(dim=1)
        a_scale = (torch.log1p(n_v.to(msgs.dtype)) / math.log1p(self.k)).unsqueeze(-1)
        combined = torch.cat([a_robust, a_spread, a_scale * a_robust], dim=-1)
        return self.project(combined)


class RobustAggregator(nn.Module):
    """Dispatches to mean / trimmed / soft_medoid aggregation.

    For trimmed and soft_medoid, optionally adds the multi-aggregator readout.
    """

    def __init__(
        self,
        d_h: int,
        aggregation: str = "trimmed",
        beta: float = 0.20,
        soft_medoid_temp: float = 1.0,
        multi_aggregator: bool = True,
        k: int = 32,
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.beta = beta
        self.soft_medoid_temp = soft_medoid_temp
        self.multi = (
            MultiAggregator(d_h, beta=beta, k=k)
            if multi_aggregator and aggregation in ("trimmed", "soft_medoid")
            else None
        )

    def forward(self, msgs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return mean_aggregate(msgs, weights)
        if self.aggregation == "trimmed":
            agg = trimmed_mean(msgs, weights, self.beta)
        elif self.aggregation == "soft_medoid":
            agg = soft_medoid(msgs, weights, self.soft_medoid_temp)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        if self.multi is not None:
            return self.multi(msgs, weights, agg)
        return agg

    def forward_batched(
        self, msgs: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Batched `forward`: every node's aggregate in one pass.

        msgs [N,K,d_h], weights [N,K], mask [N,K] -> [N,d_h]. Nodes with no
        valid neighbours yield an all-zero row, matching the loop version's
        `if not m.any(): continue` (which left `out[v]` at its zero init).
        """
        if self.aggregation == "mean":
            return mean_aggregate_batched(msgs, weights, mask)

        spread = None
        if self.aggregation == "trimmed":
            # One sort serves both the aggregate and the multi-aggregator's
            # spread term — see trimmed_mean_std_batched (memory, not style).
            agg, spread = trimmed_mean_std_batched(
                msgs, weights, mask, self.beta, want_std=self.multi is not None
            )
        elif self.aggregation == "soft_medoid":
            agg = soft_medoid_batched(msgs, weights, mask, self.soft_medoid_temp)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if self.multi is not None:
            # `spread` is None for soft_medoid (different beta bookkeeping) —
            # forward_batched recomputes it in that case.
            agg = self.multi.forward_batched(msgs, weights, mask, agg, spread=spread)
        # Isolated nodes contribute nothing (projection bias would otherwise
        # leak a constant into rows the loop version left at exactly zero).
        return agg.masked_fill(~mask.any(dim=1, keepdim=True), 0.0)
