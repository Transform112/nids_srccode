"""Additive-margin softmax (AM-Softmax) and compactness losses.

See docs/06_TRAINING.md §2.1, §2.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """AM-Softmax with optional scale warmup and class weights."""

    def __init__(
        self,
        margin: float = 0.35,
        scale_start: float = 10.0,
        scale_final: float = 30.0,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.scale_start = scale_start
        self.scale_final = scale_final
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing

    def get_scale(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.scale_final
        t = min(1.0, epoch / self.warmup_epochs)
        return self.scale_start + t * (self.scale_final - self.scale_start)

    def forward(
        self,
        cos_c: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Args:
            cos_c: [B, C] cosine to class prototypes
            targets: [B] long
            class_weights: [C]
            epoch: current epoch for scale warmup
        """
        s = self.get_scale(epoch)
        logits = s * (cos_c - self.margin * F.one_hot(targets, num_classes=cos_c.shape[1]).float())
        return F.cross_entropy(
            logits, targets, weight=class_weights, label_smoothing=self.label_smoothing
        )


class CompactnessLoss(nn.Module):
    """Explicit compactness on the hypersphere."""

    def forward(
        self,
        z: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            z: [B, d_z]
            prototypes: [C, d_z]
            targets: [B]
        """
        target_proto = prototypes[targets]
        return (1.0 - (z * target_proto).sum(dim=1)).mean()
