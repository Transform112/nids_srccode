"""Evidential Prototype Classifier (EPC) head and fallback heads.

See docs/05_ARCHITECTURE.md §6.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.models.prototypes import PrototypeBank


class EPCEmbedding(nn.Module):
    """Project edge representation to unit-norm prototype space."""

    def __init__(self, d_h: int, d_z: int) -> None:
        super().__init__()
        self.W_z = nn.Linear(d_h, d_z, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.W_z(h)
        return F.normalize(z, dim=-1)


class EPCHead(nn.Module):
    """Log-space Dirichlet evidence head with three-way decision."""

    def __init__(
        self,
        d_h: int,
        d_z: int,
        class_names: list[str],
        class_counts: dict[str, int] | None = None,
        sub_prototypes_benign: int = 4,
        sub_prototypes_attack_large: int = 2,
        sub_prototypes_attack_small: int = 1,
        margin_m: float = 0.35,
        tau_start: float = 1.0,
        tau_final: float = 0.10,
        tau_anneal_epochs: int = 6,
        tau_min: float = 0.02,
        log_evidence_clamp: float = 15.0,
        fp32_head: bool = True,
        theta_unknown: float | None = None,
        theta_defer: float | None = None,
    ) -> None:
        super().__init__()
        self.embedding = EPCEmbedding(d_h, d_z)
        self.prototype_bank = PrototypeBank(
            d_z=d_z,
            class_names=class_names,
            class_counts=class_counts,
            sub_prototypes_benign=sub_prototypes_benign,
            sub_prototypes_attack_large=sub_prototypes_attack_large,
            sub_prototypes_attack_small=sub_prototypes_attack_small,
        )
        self.d_z = d_z
        self.margin_m = margin_m
        self.tau_start = tau_start
        self.tau_final = tau_final
        self.tau_anneal_epochs = tau_anneal_epochs
        self.tau_min = tau_min
        self.log_evidence_clamp = log_evidence_clamp
        self.fp32_head = fp32_head
        self.theta_unknown = theta_unknown
        self.theta_defer = theta_defer
        # tau as learnable parameter, constrained to > tau_min
        tau0 = tau_start
        tau_hat = math.log(math.expm1(tau0 - tau_min))
        self.tau_hat = nn.Parameter(torch.tensor(tau_hat))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_hat) + self.tau_min

    def anneal_tau(self, epoch: int) -> None:
        """Set tau_hat so tau follows geometric schedule from tau_start to tau_final."""
        if self.tau_anneal_epochs <= 0:
            return
        ratio = min(1.0, epoch / self.tau_anneal_epochs)
        target = self.tau_start * (self.tau_final / self.tau_start) ** ratio
        target = max(target, self.tau_min + 1e-6)
        tau_hat = math.log(math.expm1(target - self.tau_min))
        with torch.no_grad():
            self.tau_hat.fill_(tau_hat)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            h: [B, d_h] edge representations
        Returns:
            dict with keys:
                z, log_e, alpha, S, log_S, p_hat, log_p_hat, vacuity,
                evidence_total, logits, cos_c, d_c
        """
        if self.fp32_head:
            with torch.autocast(device_type=h.device.type, enabled=False):
                return self._forward_fp32(h.float())
        return self._forward_fp32(h)

    def _forward_fp32(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.embedding(h)  # [B, d_z]
        cos_c = self.prototype_bank.cosine_to_classes(z)  # [B, C]
        d_c = 1.0 - cos_c  # [B, C]
        m = self.margin_m
        tau = self.tau
        # log e_c = -(d_c - m) / tau
        log_e = -(d_c - m) / tau.clamp_min(self.tau_min)
        log_e = torch.clamp(log_e, -self.log_evidence_clamp, self.log_evidence_clamp)
        e = torch.exp(log_e)
        c = e.shape[1]
        # log S = logsumexp([log e_1, ..., log e_C, log C])
        log_c = math.log(max(c, 1))
        log_S = torch.logsumexp(torch.cat([log_e, torch.full_like(log_e[:, :1], log_c)], dim=1), dim=1)
        S = torch.exp(log_S)
        alpha = e + 1.0
        log_alpha = torch.log1p(e)
        # log p_hat = log(alpha) - log(S), log vacuity = log(C) - log(S)
        # (docs/05_ARCHITECTURE.md §6.3 rule 2) — derive the real-valued
        # p_hat/vacuity from these via one final exp(), rather than dividing
        # in real space and then re-logging the result for log_p_hat/logits.
        # Mathematically equivalent at log_evidence_clamp=15, but avoids
        # reintroducing the exp()-mediated gradient path the log-space
        # mandate exists to avoid if the clamp or tau_min is ever loosened.
        log_p_hat = log_alpha - log_S.unsqueeze(-1)
        log_vacuity = log_c - log_S
        p_hat = torch.exp(log_p_hat)
        vacuity = torch.exp(log_vacuity)
        return {
            "z": z,
            "log_e": log_e,
            "alpha": alpha,
            "S": S,
            "log_S": log_S,
            "p_hat": p_hat,
            "log_p_hat": log_p_hat,
            "vacuity": vacuity,
            "evidence_total": e.sum(dim=1),
            "logits": log_p_hat,
            "cos_c": cos_c,
            "d_c": d_c,
        }

    def decide(
        self,
        outputs: dict[str, torch.Tensor],
        theta_unknown: float | None = None,
        theta_defer: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Three-way decision rule.

        Returns:
            decisions: [B] int (class index, or -1 for UNKNOWN, or -2 for DEFER)
            confidences: [B]
        """
        theta_unknown = theta_unknown if theta_unknown is not None else (
            self.theta_unknown if self.theta_unknown is not None else 0.5
        )
        theta_defer = theta_defer if theta_defer is not None else (
            self.theta_defer if self.theta_defer is not None else 0.1
        )
        p = outputs["p_hat"]
        E_total = outputs["evidence_total"]
        top2 = torch.topk(p, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        pred = p.argmax(dim=1)
        decisions = pred.clone()
        decisions[E_total < theta_unknown] = -1
        decisions[(E_total >= theta_unknown) & (margin < theta_defer)] = -2
        return decisions, margin

    def register_class(self, name: str, h: torch.Tensor, n_sub: int = 1) -> int:
        """Gradient-free few-shot class registration."""
        self.eval()
        with torch.no_grad():
            z = self.embedding(h)
            return self.prototype_bank.register_class(name, z, n_sub=n_sub)


class SoftmaxHead(nn.Module):
    """Baseline softmax head for ablations."""

    def __init__(self, d_h: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_h, num_classes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.fc(h)
        return {"logits": logits, "p_hat": F.softmax(logits, dim=1)}


class DistanceThresholdHead(nn.Module):
    """Fallback nearest-prototype + threshold head."""

    def __init__(
        self,
        d_h: int,
        d_z: int,
        class_names: list[str],
        class_counts: dict[str, int] | None = None,
        sub_prototypes_benign: int = 4,
        sub_prototypes_attack_large: int = 2,
        sub_prototypes_attack_small: int = 1,
        theta_unknown: float | None = None,
    ) -> None:
        super().__init__()
        self.embedding = EPCEmbedding(d_h, d_z)
        self.prototype_bank = PrototypeBank(
            d_z=d_z,
            class_names=class_names,
            class_counts=class_counts,
            sub_prototypes_benign=sub_prototypes_benign,
            sub_prototypes_attack_large=sub_prototypes_attack_large,
            sub_prototypes_attack_small=sub_prototypes_attack_small,
        )
        self.theta_unknown = theta_unknown

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.embedding(h)
        cos_c = self.prototype_bank.cosine_to_classes(z)
        logits = cos_c / 0.1  # temperature scale for compatibility
        return {
            "z": z,
            "logits": logits,
            "p_hat": F.softmax(logits, dim=1),
            "cos_c": cos_c,
            "d_c": 1.0 - cos_c,
        }

    def decide(
        self, outputs: dict[str, torch.Tensor], theta_unknown: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = theta_unknown if theta_unknown is not None else (self.theta_unknown or 0.5)
        cos_c = outputs["cos_c"]
        best = cos_c.max(dim=1)
        pred = best.indices
        decisions = pred.clone()
        decisions[best.values < theta] = -1
        return decisions, best.values


def build_head(cfg, class_names: list[str], class_counts: dict[str, int] | None = None):
    """Factory for heads from config."""
    head_type = cfg.head.type
    kwargs = dict(
        d_h=cfg.model.d_h,
        d_z=cfg.model.d_z,
        class_names=class_names,
        class_counts=class_counts,
        sub_prototypes_benign=cfg.head.sub_prototypes_benign,
        sub_prototypes_attack_large=cfg.head.sub_prototypes_attack_large,
        sub_prototypes_attack_small=cfg.head.sub_prototypes_attack_small,
    )
    if head_type == "epc":
        return EPCHead(
            **kwargs,
            margin_m=cfg.head.margin_m,
            tau_start=cfg.head.tau_start,
            tau_final=cfg.head.tau_final,
            tau_anneal_epochs=cfg.head.tau_anneal_epochs,
            tau_min=cfg.head.tau_min,
            log_evidence_clamp=cfg.head.log_evidence_clamp,
            fp32_head=cfg.head.fp32_head,
            theta_unknown=cfg.head.theta_unknown,
            theta_defer=cfg.head.theta_defer,
        )
    if head_type == "softmax":
        return SoftmaxHead(cfg.model.d_h, num_classes=len(class_names))
    if head_type == "distance_threshold":
        return DistanceThresholdHead(**kwargs, theta_unknown=cfg.head.theta_unknown)
    raise ValueError(f"Unknown head type: {head_type}")
