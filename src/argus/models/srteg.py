"""Structure-Robust Temporal Edge GNN (SR-TEG) encoder.

See docs/05_ARCHITECTURE.md §3.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.models.aggregation import RobustAggregator
from argus.models.attention import TimeDecayedAttention
from argus.models.norm import make_norm
from argus.models.time_encoding import Time2Vec


def _edges_to_dense(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_dt: torch.Tensor,
    k: int,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sampled neighbour edges to dense [N, K, *] tensors.

    Args:
        edge_index: [2, E] directed edges (src, dst)
        edge_attr: [E, F]
        edge_dt: [E]
        k: neighbour cap
        n_nodes: the true total node count (must match the model's node-state
            tensor); a scale's edges may not reference every node, so this must
            never be inferred from edge_index alone.
    Returns:
        neigh_attr: [N, K, F]
        neigh_dt: [N, K]
        mask: [N, K] bool
    """
    f = edge_attr.shape[1]
    device = edge_index.device

    neigh_attr = torch.zeros(n_nodes, k, f, device=device, dtype=edge_attr.dtype)
    neigh_dt = torch.zeros(n_nodes, k, device=device, dtype=edge_dt.dtype)
    mask = torch.zeros(n_nodes, k, dtype=torch.bool, device=device)

    # Per-destination counts
    dst = edge_index[1]
    counts = torch.zeros(n_nodes, dtype=torch.long, device=device)
    counts.scatter_add_(0, dst, torch.ones_like(dst))
    counts = counts.clamp_max(k)

    # Fill dense tensor by iterating over destination nodes.
    # This is O(N*K) and fast enough for K=32 and N up to a batch.
    for v in range(n_nodes):
        e_idx = (dst == v).nonzero(as_tuple=True)[0]
        c = counts[v].item()
        if c == 0:
            continue
        c = min(c, k)
        keep = e_idx[:c]
        neigh_attr[v, :c] = edge_attr[keep]
        neigh_dt[v, :c] = edge_dt[keep]
        mask[v, :c] = True
    return neigh_attr, neigh_dt, mask


class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch)."""

    def __init__(self, p: float = 0.05) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, device=x.device).bernoulli_(keep) / keep
        return x * mask


class SRTEGLayer(nn.Module):
    """One SR-TEG message-passing layer.

    Supports dense per-node neighbour tensors for easy trimming / soft-medoid.
    """

    def __init__(
        self,
        d_h: int,
        d_t: int,
        heads: int = 4,
        scale_duration: float = 1.0,
        aggregation: str = "trimmed",
        beta: float = 0.20,
        soft_medoid_temp: float = 1.0,
        multi_aggregator: bool = True,
        k: int = 32,
        dropout: float = 0.1,
        droppath: float = 0.05,
        norm_node: str = "graphnorm",
        norm_mlp: str = "layernorm",
        te4_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.d_t = d_t
        self.te4_enabled = te4_enabled
        self.attn = TimeDecayedAttention(d_h, heads=heads, scale_duration=scale_duration)
        self.aggregator = RobustAggregator(
            d_h=d_h,
            aggregation=aggregation,
            beta=beta,
            soft_medoid_temp=soft_medoid_temp,
            multi_aggregator=multi_aggregator,
            k=k,
        )

        # MLP_msg: d_h + d_h + d_t -> d_h, hidden 2*d_h
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_h + d_h + d_t, 2 * d_h),
            make_norm(2 * d_h, norm_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_h, d_h),
        )

        # MLP_upd: 2*d_h -> d_h, hidden 2*d_h
        self.upd_norm = make_norm(2 * d_h, norm_node)
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * d_h, 2 * d_h),
            make_norm(2 * d_h, norm_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_h, d_h),
        )
        self.drop_path = DropPath(droppath)
        # Opt-in side channel for the "attention weights alone" XAI baseline
        # (docs/11_XAI.md §2). Off by default; does not affect forward's return
        # value, numerics, or gradients when disabled.
        self.record_attention = False
        self.last_attn: dict[int, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        neigh_attr: torch.Tensor,
        neigh_dt: torch.Tensor,
        time_enc: torch.Tensor,
        mask: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            x: [N, d_h] node states
            neigh_attr: [N, K, F_e]
            neigh_dt: [N, K] time diffs normalised by scale duration
            time_enc: [N, K, d_t] precomputed Time2Vec(dt)
            mask: [N, K] bool
            batch: [N] graph ids for GraphNorm
        Returns:
            [N, d_h] updated node states
        """
        n, k, _ = neigh_attr.shape
        # Message construction: concat[src_state, edge_attr, time_enc]
        src_state = x.unsqueeze(1).expand(-1, k, -1)  # [N, K, d_h]
        msg_input = torch.cat([src_state, neigh_attr, time_enc], dim=-1)  # [N, K, d_h+d_h+d_t]
        msgs = self.msg_mlp(msg_input)  # [N, K, d_h]

        if self.record_attention:
            self.last_attn = {}

        # Per-node attention + aggregation
        out = torch.zeros_like(x)
        for v in range(n):
            m = mask[v]
            if not m.any():
                continue
            node_msgs = msgs[v, m]  # [n_v, d_h]
            query = x[v]  # [d_h]
            dt = neigh_dt[v, m]  # [n_v]
            n_v = node_msgs.shape[0]
            if self.te4_enabled:
                # Compute time-decayed attention weights using the module parameters.
                q = self.attn.W_Q(query).view(self.attn.heads, self.attn.d_k)  # [H, d_k]
                k = self.attn.W_K(node_msgs).view(n_v, self.attn.heads, self.attn.d_k)
                scores = torch.einsum("hd,nhd->nh", q, k) / math.sqrt(self.attn.d_k)
                lambdas = F.softplus(self.attn.lambda_hat)
                decay = -(lambdas.unsqueeze(0) * dt.unsqueeze(-1))
                attn = F.softmax(scores + decay, dim=0)  # [n_v, H]
                # Aggregate messages per head then concatenate.
                head_msgs = torch.einsum("nh,nhd->hd", attn, k).reshape(self.attn.d_h)
                attended = self.attn.W_O(head_msgs)
                # Derive a single weight vector per message for the robust aggregator
                # by averaging the multi-head attention weights.
                weights = attn.mean(dim=1)  # [n_v]
            else:
                attended = node_msgs.mean(dim=0)
                weights = torch.full(
                    (n_v,), 1.0 / n_v, device=x.device, dtype=x.dtype
                )
            if self.record_attention:
                self.last_attn[v] = weights.detach()
            # Robust aggregator uses raw messages with the attention-derived weights.
            # The attended value is ignored in favour of the trimmed-mean aggregate,
            # keeping the breakdown-point argument valid.
            agg = self.aggregator(node_msgs, weights)  # [d_h]
            out[v] = agg

        # Node update: pre-norm residual
        concat = torch.cat([x, out], dim=-1)
        concat = self.upd_norm(concat, batch)
        upd = self.upd_mlp(concat)
        return x + self.drop_path(upd)


class SRTEGEncoder(nn.Module):
    """Full SR-TEG encoder: per-scale edge projection + L layers + edge readout."""

    def __init__(
        self,
        f_e: int,
        f_v: int,
        d_h: int,
        d_a: int,
        d_b: int,
        d_t: int,
        layers: int = 2,
        heads: int = 4,
        k: int = 32,
        aggregation: str = "trimmed",
        beta: float = 0.20,
        soft_medoid_temp: float = 1.0,
        multi_aggregator: bool = True,
        dropout: float = 0.1,
        droppath: float = 0.05,
        norm_node: str = "graphnorm",
        norm_mlp: str = "layernorm",
        time_encoding: str = "time2vec",
        time2vec_period_min: float = 0.1,
        time2vec_period_max: float = 600.0,
        te3_enabled: bool = True,
        te4_enabled: bool = True,
        channel_a_indices: Sequence[int] | None = None,
        channel_b_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.f_e = f_e
        self.f_v = f_v
        self.d_h = d_h
        self.d_a = d_a
        self.d_b = d_b
        self.d_t = d_t
        self.layers = layers
        self.k = k
        self.te3_enabled = te3_enabled

        if channel_a_indices is None or channel_b_indices is None:
            # Default: first d_a features A, rest B. Real code uses partition.py.
            channel_a_indices = list(range(min(d_a, f_e)))
            channel_b_indices = [i for i in range(f_e) if i not in channel_a_indices]
        self.register_buffer("a_idx", torch.tensor(channel_a_indices, dtype=torch.long))
        self.register_buffer("b_idx", torch.tensor(channel_b_indices, dtype=torch.long))

        # Node projection W_v
        self.W_v = nn.Linear(f_v, d_h, bias=True)

        # Edge channel projections
        self.W_A = nn.Linear(len(channel_a_indices), d_a, bias=True)
        self.W_B = nn.Linear(len(channel_b_indices), d_b, bias=True)
        self.norm_A = make_norm(d_a, norm_mlp)
        self.norm_B = make_norm(d_b, norm_mlp)

        # Time encoding
        if time_encoding == "time2vec":
            self.time_enc = Time2Vec(d_t, period_min=time2vec_period_min, period_max=time2vec_period_max)
        else:
            from argus.models.time_encoding import RecencyOnlyEncoding
            self.time_enc = RecencyOnlyEncoding(d_t)

        # Scale-specific embedding epsilon_s
        self.scale_eps = nn.Parameter(torch.zeros(3, d_h))

        # GNN layers (weight-shared across scales handled by forward loop)
        self.gnn_layers = nn.ModuleList(
            [
                SRTEGLayer(
                    d_h=d_h,
                    d_t=d_t,
                    heads=heads,
                    scale_duration=1.0,  # overridden per scale in forward
                    aggregation=aggregation,
                    beta=beta,
                    soft_medoid_temp=soft_medoid_temp,
                    multi_aggregator=multi_aggregator,
                    k=k,
                    dropout=dropout,
                    droppath=droppath,
                    norm_node=norm_node,
                    norm_mlp=norm_mlp,
                    te4_enabled=te4_enabled,
                )
                for _ in range(layers)
            ]
        )

        # Edge readout MLP
        self.edge_readout = nn.Sequential(
            nn.Linear(3 * d_h + d_t, 2 * d_h),
            make_norm(2 * d_h, norm_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_h, d_h),
        )

    def _project_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Provenance-partitioned edge projection."""
        x_a = edge_attr[..., self.a_idx]
        x_b = edge_attr[..., self.b_idx]
        u_a = F.gelu(self.norm_A(self.W_A(x_a)))
        u_b = F.gelu(self.norm_B(self.W_B(x_b)))
        return torch.cat([u_a, u_b], dim=-1)

    def forward_scale(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_dt: torch.Tensor,
        scale_duration: float = 1.0,
        memory: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        scale_id: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder for one scale.

        Args:
            node_feat: [N, F_v]
            edge_index: [2, E]
            edge_attr: [E, F_e]
            edge_dt: [E] seconds, already normalised or not
            scale_duration: D_s for time encoding / decay init
            memory: [N, d_h] per-node memory to add to initial states
            batch: [N] graph ids
            scale_id: 0=short, 1=mid, 2=long
        Returns:
            x: [N, d_h] final node states
            edge_repr: [E, d_h] edge representations
            long_aggregate: [N, d_h] aggregate from the last layer (for memory update)
        """
        device = node_feat.device
        dtype = node_feat.dtype

        # Initial node states
        x = self.W_v(node_feat)
        if memory is not None:
            x = x + memory

        u_e = self._project_edges(edge_attr)
        u_e = u_e + self.scale_eps[scale_id]

        neigh_attr, neigh_dt, mask = _edges_to_dense(edge_index, u_e, edge_dt, self.k, n_nodes=x.shape[0])
        if self.te3_enabled:
            time_enc = self.time_enc(neigh_dt.view(-1), scale_duration).view(
                *neigh_dt.shape, self.d_t
            )
        else:
            from argus.models.time_encoding import RecencyOnlyEncoding
            enc = RecencyOnlyEncoding(self.d_t)
            time_enc = enc(neigh_dt.view(-1), scale_duration).view(*neigh_dt.shape, self.d_t)

        # GNN layers
        long_agg = None
        for layer in self.gnn_layers:
            layer.attn.scale_duration = scale_duration
            x = layer(x, neigh_attr, neigh_dt, time_enc, mask, batch=batch)
            # For memory we need the last-layer aggregate on the long scale.
            if scale_id == 2:
                long_agg = x  # placeholder; real aggregate requires recomputation

        # Edge readout
        src, dst = edge_index[0], edge_index[1]
        h_u, h_v = x[src], x[dst]
        zero_dt = torch.zeros(edge_dt.shape[0], device=device, dtype=dtype)
        phi0 = self.time_enc(zero_dt, scale_duration)
        edge_repr = self.edge_readout(torch.cat([h_u, h_v, u_e, phi0], dim=-1))
        return x, edge_repr, long_agg if long_agg is not None else x

    def encode_target_edges(
        self,
        x: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
        scale_duration: float = 1.0,
        scale_id: int = 1,
    ) -> torch.Tensor:
        """Compute edge representations for target edges using already-computed
        node states `x` from this scale's message passing.

        Target edges need not be part of the context edge set used for message
        passing (docs/04_GRAPH_CONSTRUCTION.md §5: "target flows are always
        present in the short-scale graph", but their mid/long-scale
        representations are derived from that scale's node states, not from a
        literal edge in that scale's sampled context).
        """
        device = x.device
        dtype = x.dtype
        u_e = self._project_edges(target_edge_attr) + self.scale_eps[scale_id]
        src, dst = target_edge_index[0], target_edge_index[1]
        h_u, h_v = x[src], x[dst]
        zero_dt = torch.zeros(target_edge_attr.shape[0], device=device, dtype=dtype)
        phi0 = self.time_enc(zero_dt, scale_duration)
        return self.edge_readout(torch.cat([h_u, h_v, u_e, phi0], dim=-1))
