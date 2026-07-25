"""Time-decayed multi-head attention.

See docs/05_ARCHITECTURE.md §3.4.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDecayedAttention(nn.Module):
    """Multi-head attention with additive log-space time decay.

    Input: query [d_h], messages [n, d_h], time_diffs [n] (normalised).
    Output: aggregated message [d_h].
    """

    def __init__(self, d_h: int, heads: int = 4, scale_duration: float = 1.0) -> None:
        super().__init__()
        if d_h % heads != 0:
            raise ValueError("d_h must be divisible by heads")
        self.d_h = d_h
        self.heads = heads
        self.d_k = d_h // heads
        self.scale_duration = scale_duration
        self.W_Q = nn.Linear(d_h, d_h, bias=False)
        self.W_K = nn.Linear(d_h, d_h, bias=False)
        self.W_O = nn.Linear(d_h, d_h, bias=False)

        # Decay rates per head, initialised for a half-life ladder.
        # Defaults: [D_s/16, D_s/8, D_s/4, D_s/2] => lambda = ln(2) / half_life
        half_lives = torch.tensor(
            [scale_duration / 16.0, scale_duration / 8.0, scale_duration / 4.0, scale_duration / 2.0]
        )
        lambdas = math.log(2.0) / half_lives.clamp_min(1e-6)
        # softplus inverse: lambda = log(exp(lambda_hat) - 1) => lambda_hat = log(exp(lambda) - 1)
        lambda_hat = torch.log(torch.expm1(lambdas))
        self.lambda_hat = nn.Parameter(lambda_hat[:heads])

    def forward(
        self,
        query: torch.Tensor,
        msgs: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            query: [d_h]
            msgs: [n, d_h]
            dt: [n] time differences, normalised by scale duration.
        Returns:
            [d_h]
        """
        n = msgs.shape[0]
        q = self.W_Q(query)  # [d_h]
        k = self.W_K(msgs)  # [n, d_h]

        q = q.view(self.heads, self.d_k)  # [H, d_k]
        k = k.view(n, self.heads, self.d_k)  # [n, H, d_k]

        scores = torch.einsum("hd,nhd->nh", q, k) / math.sqrt(self.d_k)  # [n, H]

        # Additive log-space decay: delta = -lambda * dt
        lambdas = F.softplus(self.lambda_hat)  # [H]
        decay = -(lambdas.unsqueeze(0) * dt.unsqueeze(-1))  # [n, H]
        attn = F.softmax(scores + decay, dim=0)  # [n, H]

        # Weighted sum per head, then project
        out = torch.einsum("nh,nhd->hd", attn, k)  # [H, d_k]
        out = out.reshape(self.d_h)
        return self.W_O(out)
