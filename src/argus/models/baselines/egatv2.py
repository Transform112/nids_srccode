"""EGATv2 / E-ResGAT baseline: constant node init, GATv2 attention over
*uncapped* neighbourhoods, residual connections, 2 layers, 4 heads, softmax
head. Single scale (mid), no time decay.

Reproduces the GATv2-attention family (Chang & Branco, arXiv:2111.13597;
Brody et al. GATv2, arXiv:2105.14491) as closely as practical for a NIDS
edge-classification baseline. See docs/05_ARCHITECTURE.md §8.

Deterministic scatter ops (`index_add_`) are used throughout instead of PyG's
`MessagePassing`, matching the convention already used by
`models/baselines/egraphsage.py` and `models/srteg.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax of `scores` [E, H] grouped by destination `index` [E], per head.

    Deterministic: uses `scatter_reduce_(reduce="amax")` for the max-shift and
    `index_add_` for the sum, avoiding PyG's non-deterministic scatter kernels.
    """
    heads = scores.shape[1]
    max_per_dst = torch.full(
        (dim_size, heads), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    max_per_dst.scatter_reduce_(
        0, index.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True
    )
    # Nodes with no incoming edges keep -inf; replace with 0 to avoid NaN (unused downstream).
    max_per_dst = torch.where(torch.isinf(max_per_dst), torch.zeros_like(max_per_dst), max_per_dst)
    shifted = scores - max_per_dst.index_select(0, index)
    exp_scores = shifted.exp()
    sum_per_dst = torch.zeros((dim_size, heads), device=scores.device, dtype=scores.dtype)
    sum_per_dst.index_add_(0, index, exp_scores)
    return exp_scores / sum_per_dst.index_select(0, index).clamp_min(1e-12)


class EGATv2Layer(nn.Module):
    """One GATv2-attention layer with a residual connection.

    Message content m_ij = Linear(concat(x_src, edge_attr_proj)) (the "value").
    Attention logit (GATv2 order — nonlinearity *before* the `a` projection):
        e_ij = a_h^T LeakyReLU(W_q x_dst + m_ij)
    Aggregation: softmax over incoming edges per destination node, per head.
    """

    def __init__(self, d_h: int, heads: int = 4) -> None:
        super().__init__()
        if d_h % heads != 0:
            raise ValueError("d_h must be divisible by heads")
        self.heads = heads
        self.d_head = d_h // heads
        self.d_h = d_h
        self.msg_lin = nn.Linear(2 * d_h, d_h)
        self.dst_lin = nn.Linear(d_h, d_h)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.att = nn.Parameter(torch.empty(heads, self.d_head))
        nn.init.xavier_uniform_(self.att)
        self.out_proj = nn.Linear(d_h, d_h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr_proj: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        msg = self.msg_lin(torch.cat([x[src], edge_attr_proj], dim=-1))  # [E, d_h]
        q = self.dst_lin(x[dst])  # [E, d_h]

        combined = self.leaky_relu(q + msg).view(-1, self.heads, self.d_head)
        scores = torch.einsum("ehd,hd->eh", combined, self.att)  # [E, heads]
        alpha = _segment_softmax(scores, dst, dim_size=n)  # [E, heads]

        weighted = (msg.view(-1, self.heads, self.d_head) * alpha.unsqueeze(-1)).reshape(-1, self.d_h)
        agg = torch.zeros(n, self.d_h, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, weighted)

        return F.elu(x + self.out_proj(agg))  # residual connection


class EGATv2(nn.Module):
    """EGATv2: constant node features, 2 GATv2-attention layers (4 heads,
    residual), softmax head. Same node-init and readout convention as
    `EGraphSAGE` so the two GNN baselines are directly comparable.
    """

    def __init__(self, f_e: int, d_h: int, num_classes: int, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.d_h = d_h
        self.node_init = nn.Parameter(torch.ones(1, d_h) * 0.1, requires_grad=False)
        self.edge_proj = nn.Linear(f_e, d_h)
        self.gnn_layers = nn.ModuleList([EGATv2Layer(d_h, heads=heads) for _ in range(layers)])
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
