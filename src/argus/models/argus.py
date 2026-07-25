"""Top-level ARGUS model: SR-TEG encoder + multi-scale fusion + head."""

from __future__ import annotations

import torch
import torch.nn as nn

from argus.models.epc import build_head
from argus.models.multiscale import MultiScaleFusion
from argus.models.srteg import SRTEGEncoder
from argus.models.memory import NodeMemory


class ArgusModel(nn.Module):
    """End-to-end ARGUS model.

    The model consumes a batch dictionary produced by the graph builder and
    outputs per-target-edge predictions. For unit tests it also accepts a
    simplified tuple of tensors.
    """

    def __init__(
        self,
        cfg,
        f_e: int,
        f_v: int,
        class_names: list[str],
        class_counts: dict[str, int] | None = None,
        channel_a_indices: list[int] | None = None,
        channel_b_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.f_e = f_e
        self.f_v = f_v
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.encoder = SRTEGEncoder(
            f_e=f_e,
            f_v=f_v,
            d_h=cfg.model.d_h,
            d_a=cfg.model.d_A,
            d_b=cfg.model.d_B,
            d_t=cfg.model.d_t,
            layers=cfg.model.layers,
            heads=cfg.model.heads,
            k=cfg.graph.neighbour_cap,
            aggregation=cfg.model.aggregation,
            beta=cfg.model.trim_beta,
            soft_medoid_temp=cfg.model.soft_medoid_temp,
            multi_aggregator=cfg.model.multi_aggregator,
            dropout=cfg.regularisation.dropout,
            droppath=cfg.regularisation.droppath,
            norm_node=cfg.model.norm_node,
            norm_mlp=cfg.model.norm_mlp,
            time_encoding=cfg.model.time_encoding,
            time2vec_period_min=cfg.model.time2vec_period_min,
            time2vec_period_max=cfg.model.time2vec_period_max,
            te3_enabled=cfg.model.te3_enabled,
            te4_enabled=cfg.model.te4_enabled,
            channel_a_indices=channel_a_indices,
            channel_b_indices=channel_b_indices,
        )
        self.memory = (
            NodeMemory(
                d_h=cfg.model.d_h,
                dropout=cfg.regularisation.memory_dropout,
                norm=cfg.model.norm_mlp,
            )
            if cfg.model.te6_enabled
            else None
        )
        self.fusion = MultiScaleFusion(
            d_h=cfg.model.d_h,
            dropout=cfg.regularisation.dropout,
            norm_mlp=cfg.model.norm_mlp,
            te5_enabled=cfg.model.te5_enabled,
        )
        self.head = build_head(cfg, class_names, class_counts)

        self.scale_durations = [
            cfg.graph.window_short_seconds,
            cfg.graph.window_mid_seconds,
            cfg.graph.window_long_seconds,
        ]

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index_s: torch.Tensor,
        edge_attr_s: torch.Tensor,
        edge_dt_s: torch.Tensor,
        edge_index_m: torch.Tensor,
        edge_attr_m: torch.Tensor,
        edge_dt_m: torch.Tensor,
        edge_index_l: torch.Tensor,
        edge_attr_l: torch.Tensor,
        edge_dt_l: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
        memory: dict[int, torch.Tensor] | None = None,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for one batch.

        Args:
            node_feat: [N, F_v]
            edge_index_*: [2, E_*] context edges used for message passing at each scale
            edge_attr_*: [E_*, F_e]
            edge_dt_*: [E_*] seconds
            target_edge_index: [2, T] (src, dst) node indices of the flows to classify,
                in the same unified node index space as edge_index_*
            target_edge_attr: [T, F_e] the target flows' own feature vectors
            memory: optional per-node memory dict
            batch: [N] graph ids
        Returns:
            outputs dict from head plus intermediate tensors.
        """
        mem_tensor = None
        if memory is not None and self.memory is not None:
            node_ids = torch.arange(node_feat.shape[0], device=node_feat.device)
            # memory not used here; integrate externally for streaming
            mem_tensor = None

        # Short scale
        x_s, _, _ = self.encoder.forward_scale(
            node_feat, edge_index_s, edge_attr_s, edge_dt_s,
            scale_duration=self.scale_durations[0],
            memory=mem_tensor, batch=batch, scale_id=0,
        )
        # Mid scale
        x_m, _, _ = self.encoder.forward_scale(
            node_feat, edge_index_m, edge_attr_m, edge_dt_m,
            scale_duration=self.scale_durations[1],
            memory=mem_tensor, batch=batch, scale_id=1,
        )
        # Long scale
        x_l, _, long_agg = self.encoder.forward_scale(
            node_feat, edge_index_l, edge_attr_l, edge_dt_l,
            scale_duration=self.scale_durations[2],
            memory=mem_tensor, batch=batch, scale_id=2,
        )

        # Update memory on long scale if enabled
        if self.memory is not None and memory is not None and long_agg is not None:
            node_ids = torch.arange(node_feat.shape[0], device=node_feat.device)
            _, memory = self.memory(node_ids, long_agg, memory)

        # Compute the target edges' representation at each scale from that
        # scale's node states (not by indexing into that scale's own context
        # edge set, which may not literally contain the target edges).
        h_target = self.encoder.encode_target_edges(
            x_s, target_edge_index, target_edge_attr, self.scale_durations[0], scale_id=0
        )
        h_mid_target = self.encoder.encode_target_edges(
            x_m, target_edge_index, target_edge_attr, self.scale_durations[1], scale_id=1
        )
        h_long_target = self.encoder.encode_target_edges(
            x_l, target_edge_index, target_edge_attr, self.scale_durations[2], scale_id=2
        )
        h_fused, gate = self.fusion(h_target, h_mid_target, h_long_target)

        head_out = self.head(h_fused)
        head_out["h_fused"] = h_fused
        head_out["gate"] = gate
        return head_out

    def register_class(self, name: str, h: torch.Tensor, n_sub: int = 1) -> int:
        """Gradient-free class registration."""
        return self.head.register_class(name, h, n_sub=n_sub)
