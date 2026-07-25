"""E-GraphSAGE baseline: constant node init, mean aggregation over uncapped
neighbourhoods, softmax head. Single scale (mid), no time encoding.

Reproduces Lo et al., NOMS 2022 (arXiv:2103.16329) as closely as practical.
See docs/05_ARCHITECTURE.md §8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Deterministic scatter-mean via index_add_ (avoids PyG's non-deterministic kernels)."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp_min(1.0).unsqueeze(-1)


class EGraphSAGELayer(nn.Module):
    """One mean-aggregation SAGE layer: concat(self, mean(neighbour msgs)) -> MLP."""

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(nn.Linear(d_h + d_h, d_h), nn.ReLU())
        self.upd_mlp = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr_proj: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], edge_attr_proj], dim=-1)
        msgs = self.msg_mlp(msg_input)
        agg = _scatter_mean(msgs, dst, dim_size=x.shape[0])
        return self.upd_mlp(torch.cat([x, agg], dim=-1))


class EGraphSAGE(nn.Module):
    """E-GraphSAGE: constant node features, 2 mean-aggregation layers, softmax head."""

    def __init__(self, f_e: int, d_h: int, num_classes: int, layers: int = 2) -> None:
        super().__init__()
        self.d_h = d_h
        self.node_init = nn.Parameter(torch.ones(1, d_h) * 0.1, requires_grad=False)
        self.edge_proj = nn.Linear(f_e, d_h)
        self.gnn_layers = nn.ModuleList([EGraphSAGELayer(d_h) for _ in range(layers)])
        self.readout = nn.Linear(2 * d_h + d_h, num_classes)  # h_u, h_v, edge_attr_proj

    def forward(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        device = edge_attr.device
        x = self.node_init.to(device).expand(n_nodes, -1).clone()
        u_e = F.relu(self.edge_proj(edge_attr))
        for layer in self.gnn_layers:
            x = layer(x, edge_index, u_e)

        src, dst = target_edge_index[0], target_edge_index[1]
        target_u_e = F.relu(self.edge_proj(target_edge_attr))
        logits = self.readout(torch.cat([x[src], x[dst], target_u_e], dim=-1))
        return logits
