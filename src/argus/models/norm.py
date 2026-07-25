"""Normalisation layers. BatchNorm is intentionally absent.

See docs/05_ARCHITECTURE.md §9.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    """GraphNorm: per-graph mean subtraction with learnable shift.

    Implements Cai et al., "GraphNorm: A principled approach to accelerating graph neural
    network training", NeurIPS 2021. Normalisation is per-graph, never per-batch.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            x: [N, d]
            batch: [N] integer graph ids; if None, treat all nodes as one graph.
        """
        if batch is None:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            return self.scale * (x - mean - self.shift) / torch.sqrt(var + self.eps)

        # Per-graph mean
        num_graphs = int(batch.max().item()) + 1
        mean = torch.zeros_like(x)
        var = torch.zeros_like(x)
        for g in range(num_graphs):
            mask = batch == g
            if mask.any():
                mean_g = x[mask].mean(dim=0, keepdim=True)
                var_g = x[mask].var(dim=0, unbiased=False, keepdim=True)
                mean[mask] = mean_g
                var[mask] = var_g
        return self.scale * (x - mean - self.shift) / torch.sqrt(var + self.eps)


class RMSNorm(nn.Module):
    """RMSNorm: per-sample root-mean-square normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


def make_norm(dim: int, kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "graphnorm":
        return GraphNorm(dim)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unsupported norm: {kind}. BatchNorm is not allowed.")
