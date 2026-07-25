"""Multi-scale gated fusion (TE5).

See docs/05_ARCHITECTURE.md §5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.models.norm import make_norm


class MultiScaleFusion(nn.Module):
    """Fuse three scale-specific edge representations with a learned gate."""

    def __init__(
        self,
        d_h: int,
        dropout: float = 0.1,
        norm_mlp: str = "layernorm",
        te5_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.te5_enabled = te5_enabled
        self.gate = nn.Linear(3 * d_h, 3, bias=True)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(3 * d_h, 2 * d_h),
            make_norm(2 * d_h, norm_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_h, d_h),
        )
        self.norm = make_norm(d_h, norm_mlp)

    def forward(
        self,
        h_short: torch.Tensor,
        h_mid: torch.Tensor,
        h_long: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            h_*: [E, d_h]
        Returns:
            h_fused: [E, d_h]
            gate_probs: [E, 3]
        """
        if not self.te5_enabled:
            return h_mid, None
        stacked = torch.cat([h_short, h_mid, h_long], dim=-1)  # [E, 3*d_h]
        gate_probs = F.softmax(self.gate(stacked), dim=-1)  # [E, 3]
        h = (
            gate_probs[:, 0:1] * h_short
            + gate_probs[:, 1:2] * h_mid
            + gate_probs[:, 2:3] * h_long
        )
        h = self.norm(h + self.fuse_mlp(stacked))
        return h, gate_probs
