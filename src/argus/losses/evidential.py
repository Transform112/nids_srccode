"""Evidential loss, Dirichlet KL regulariser, and synthetic-unknown losses.

See docs/06_TRAINING.md §2.3, §2.5, §2.6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_kl(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(target)).

    Both alpha and target have shape [B, C] and values > 0.
    """
    return (
        torch.lgamma(alpha.sum(dim=1))
        - torch.lgamma(alpha).sum(dim=1)
        - torch.lgamma(target.sum(dim=1))
        + torch.lgamma(target).sum(dim=1)
        + ((alpha - target) * (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True)))).sum(dim=1)
    )


class EvidentialLoss(nn.Module):
    """Type-II maximum likelihood under Dirichlet posterior.

    L_evid = sum_k y_k (psi(S) - psi(alpha_k))
    """

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        kl_anneal_weight: float = 0.0,
    ) -> torch.Tensor:
        """Args:
            alpha: [B, C] Dirichlet concentrations
            targets: [B]
            kl_anneal_weight: weight for KL regulariser term
        """
        S = alpha.sum(dim=1)
        y = F.one_hot(targets, num_classes=alpha.shape[1]).float()
        loss = (y * (torch.digamma(S.unsqueeze(1)) - torch.digamma(alpha))).sum(dim=1).mean()
        if kl_anneal_weight > 0:
            target_dir = y + (1.0 - y) * alpha
            kl = dirichlet_kl(target_dir, torch.ones_like(alpha))
            loss = loss + kl_anneal_weight * kl.mean()
        return loss


class SyntheticUnknownLoss(nn.Module):
    """Drive total evidence to zero for synthetic unknowns."""

    def __init__(self, eps_floor: float = 0.01) -> None:
        super().__init__()
        self.eps_floor = eps_floor

    def forward(self, log_e: torch.Tensor) -> torch.Tensor:
        """Args:
            log_e: [B, C] log evidence for synthetic unknowns
        """
        e = torch.exp(log_e)
        total = e.sum(dim=1)
        return (total + F.relu(total - self.eps_floor)).mean()
