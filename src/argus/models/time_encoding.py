"""Time encoding modules: Time2Vec with log-grid init and Bochner fallback.

See docs/05_ARCHITECTURE.md §2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec encoding with learnable frequencies.

    phi(dt)[0] = omega_0 * dt + b_0
    phi(dt)[i] = sin(omega_i * dt + b_i)

    Frequencies initialised on a log-uniform grid covering periods from
    `period_min` to `period_max` seconds. dt is normalised by scale duration.
    """

    def __init__(
        self,
        dim: int,
        period_min: float = 0.1,
        period_max: float = 600.0,
        omega_0_init: float = 1.0 / 300.0,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("Time2Vec dim must be >= 2")
        self.dim = dim
        self.omega = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self._init_log_grid(period_min, period_max, omega_0_init)

    def _init_log_grid(self, period_min: float, period_max: float, omega_0_init: float) -> None:
        # Linear term
        self.omega.data[0] = omega_0_init
        self.bias.data[0] = 0.0

        # Sinusoidal terms: log-uniform periods
        periods = torch.logspace(
            math.log10(period_min), math.log10(period_max), self.dim - 1
        )
        freqs = 2.0 * math.pi / periods
        self.omega.data[1:] = freqs
        self.bias.data[1:].uniform_(0, 2.0 * math.pi)

    def forward(self, dt: torch.Tensor, scale_duration: float = 1.0) -> torch.Tensor:
        """Args:
            dt: [...] time difference in seconds (>=0)
            scale_duration: normalise dt by this value before encoding.
        Returns:
            [... , dim] encoding.
        """
        t = dt.unsqueeze(-1) / scale_duration  # [... , 1]
        linear = self.omega[0] * t + self.bias[0]
        sinusoidal = torch.sin(self.omega[1:] * t + self.bias[1:])
        return torch.cat([linear, sinusoidal], dim=-1)


class RecencyOnlyEncoding(nn.Module):
    """Ablation hook when TE3 is disabled: broadcast log1p(dt) to dim zeros."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, dt: torch.Tensor, scale_duration: float = 1.0) -> torch.Tensor:
        t = torch.log1p(dt.unsqueeze(-1) / scale_duration)
        zeros = torch.zeros(*dt.shape, self.dim - 1, device=dt.device, dtype=dt.dtype)
        return torch.cat([t, zeros], dim=-1)
