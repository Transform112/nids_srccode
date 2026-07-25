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


class MultiAggregator(nn.Module):
    """Concatenates robust mean, trimmed std, and degree-scaled robust mean.

    Projects back to d_h. See docs/05_ARCHITECTURE.md §3.5b.
    """

    def __init__(self, d_h: int, beta: float = 0.20, k: int = 32) -> None:
        super().__init__()
        self.d_h = d_h
        self.beta = beta
        self.k = k
        self.project = nn.Linear(2 * d_h + 1, d_h, bias=True)

    def forward(self, msgs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Args:
            msgs: [n, d_h]
            weights: [n]
        Returns:
            [d_h]
        """
        a_robust = trimmed_mean(msgs, weights, self.beta)
        a_spread = trimmed_std(msgs, weights, self.beta)
        n = msgs.shape[0]
        a_scale = math.log1p(n) / math.log1p(self.k)
        combined = torch.cat([a_robust, a_spread, torch.tensor([a_scale], device=msgs.device)], dim=0)
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
            return self.multi(msgs, weights)
        return agg
