"""Anomal-E baseline: graph-autoencoder anomaly detection.

Edge-level reconstruction: encoder compresses edge features + graph context,
decoder reconstructs the input features. Anomaly score = reconstruction MSE.
No open-set head needed — anomalies are flows that cannot be reconstructed.

See docs/05_ARCHITECTURE.md §8.2 — reproduces the general approach of Caville
et al. (Anomal-E, 2022) but applied to the edge-level NIDS setting with the
same graph batches ARGUS uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Deterministic scatter-mean via index_add_."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp_min(1.0).unsqueeze(-1)


class AnomalEEncoder(nn.Module):
    """Single-scale encoder: edge features + neighbour context → latent z."""

    def __init__(self, f_e: int, d_h: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(f_e, d_h)
        self.msg_mlp = nn.Sequential(nn.Linear(d_h + d_h, d_h), nn.ReLU())
        self.upd_mlp = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU())
        self.to_latent = nn.Linear(d_h, latent_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode edge features into latent embeddings.

        Returns:
            z_e: [E, latent_dim] per-edge latent codes.
            h: [n_nodes, d_h] node representations (for loss).
        """
        src, dst = edge_index[0], edge_index[1]
        u_e = F.relu(self.edge_proj(edge_attr))

        # Initial node feature: mean of incoming edge projections
        h = F.relu(self.edge_proj.weight.new_zeros(n_nodes, self.edge_proj.out_features))
        h.index_add_(0, dst, u_e)
        count = torch.zeros(n_nodes, device=h.device)
        count.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        h = h / count.clamp_min(1.0).unsqueeze(-1)

        # One message-passing round
        msg_input = torch.cat([h[src], u_e], dim=-1)
        msgs = self.msg_mlp(msg_input)
        agg = _scatter_mean(msgs, dst, dim_size=n_nodes)
        h = F.relu(self.upd_mlp(torch.cat([h, agg], dim=-1)))

        z_e = self.to_latent(h[src])
        return z_e, h


class AnomalEDecoder(nn.Module):
    """Reconstruct edge features from latent z."""

    def __init__(self, latent_dim: int, f_e: int, d_h: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, f_e),
        )

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return self.net(z_e)


class AnomalE(nn.Module):
    """Graph autoencoder for anomaly detection.

    Trained to minimise MSE between input edge features and their reconstructions.
    At test time, the reconstruction error is the anomaly score.
    """

    def __init__(
        self,
        f_e: int,
        d_h: int = 128,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = AnomalEEncoder(f_e, d_h, latent_dim)
        self.decoder = AnomalEDecoder(latent_dim, f_e, d_h)
        self.f_e = f_e

    def forward(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return reconstruction and anomaly scores for target edges."""
        z_e, _ = self.encoder(edge_index, edge_attr, n_nodes)
        # For target edges, use the source node's latent
        src = target_edge_index[0]
        z_src = z_e[src] if src.max() < z_e.shape[0] else z_e.new_zeros(
            target_edge_attr.shape[0], z_e.shape[1]
        )
        recon = self.decoder(z_src)
        recon_error = F.mse_loss(recon, target_edge_attr, reduction="none").mean(dim=-1)
        return {
            "recon": recon,
            "anomaly_score": recon_error,
            "logits": -recon_error.unsqueeze(-1).expand(-1, 2),  # 2-class: normal/anomalous
        }

    def anomaly_score(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: return per-edge anomaly scores."""
        return self.forward(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)[
            "anomaly_score"
        ]
