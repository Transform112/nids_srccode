"""Provenance channel gradient penalty.

See docs/06_TRAINING.md §2.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelPenaltyLoss(nn.Module):
    """Penalise reliance on Channel A relative to total gradient norm.

    Operates on a single edge-feature tensor (as produced by the graph batcher)
    plus the channel-A / channel-B column index tensors from the encoder, so it
    can be applied without threading two separate leaf tensors through the model.
    """

    def __init__(self, rho: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        self.rho = rho
        self.eps = eps

    def forward(
        self,
        loss: torch.Tensor,
        edge_attr: torch.Tensor,
        a_idx: torch.Tensor,
        b_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            loss: scalar loss to differentiate (graph must still be alive)
            edge_attr: [B, F_e] edge features with requires_grad=True, used as
                model input for the forward pass that produced `loss`
            a_idx: [|A|] column indices of the controllable channel
            b_idx: [|B|] column indices of the observer channel
        Returns:
            scalar penalty
        """
        (grad,) = torch.autograd.grad(
            loss,
            [edge_attr],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        if grad is None:
            return torch.tensor(0.0, device=edge_attr.device)
        g_a_norm = grad[:, a_idx].norm(dim=1)
        g_b_norm = grad[:, b_idx].norm(dim=1)
        denom = g_a_norm + g_b_norm + self.eps
        ratio = g_a_norm / denom
        ratio = ratio.clamp(0.0, 1.0)
        return torch.relu(ratio - self.rho).pow(2).mean()
