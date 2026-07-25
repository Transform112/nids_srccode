"""A3 — prototype poisoning.

See docs/10_ADVERSARIAL.md §4. Applicable only when streaming EMA prototype
drift correction is enabled (`prototype_ema_momentum` not `off`;
`models/prototypes.py::PrototypeBank.ema_update`). Salvages the memory-
poisoning idea from the project's earlier TGN iteration (`plan/previous_work.txt`).

Operates directly at the embedding level (post `EPCHead.embedding`), since the
mechanism under test is the prototype bank + evidence gate, not the encoder —
consistent with the doc's framing of A3 as a *streaming* attack on
already-computed, already-accepted embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from argus.models.epc import EPCHead


@dataclass
class A3Result:
    poison_rate: float
    gate_enabled: bool
    final_drift: float
    accepted_fraction: float
    drift_curve: list[float]


def craft_poison_embedding(
    head: EPCHead, class_idx: int, edge_strength: float = 0.9, seed: int = 0
) -> torch.Tensor:
    """Sample a unit-norm embedding at the far edge of class `class_idx`'s cone:
    the class's (nearest) sub-prototype displaced by a random tangent
    perturbation, scaled by `edge_strength` (0 = exactly the prototype, higher
    = further toward the edge of the cone while still nominally belonging to it).
    """
    g = torch.Generator().manual_seed(seed)
    bank = head.prototype_bank
    mask = torch.tensor([i == class_idx for i in bank.class_of])
    proto = bank.bank.data[mask][0]  # nearest/only sub-prototype (common attack-class case)
    tangent = torch.randn(bank.d_z, generator=g)
    tangent = tangent - (tangent @ proto) * proto  # project out the radial component
    tangent = F.normalize(tangent, dim=0)
    z = proto + edge_strength * tangent
    return F.normalize(z, dim=0)


def passes_evidence_gate(head: EPCHead, z: torch.Tensor, class_idx: int, theta_unknown: float) -> bool:
    """Replicate `EPCHead`'s log-space evidence computation directly on an
    embedding `z` (bypassing the encoder + `EPCHead.embedding` projection,
    since the object under test is the prototype geometry, not the encoder).
    """
    with torch.no_grad():
        cos_c = head.prototype_bank.cosine_to_classes(z.unsqueeze(0))  # [1, C]
        d_c = 1.0 - cos_c
        tau = head.tau.clamp_min(head.tau_min)
        log_e = torch.clamp(-(d_c - head.margin_m) / tau, -head.log_evidence_clamp, head.log_evidence_clamp)
        evidence_total = float(torch.exp(log_e).sum().item())
    return evidence_total >= theta_unknown


def run_a3_poison_sweep(
    head: EPCHead,
    class_idx: int,
    poison_rate: float,
    momentum: float,
    n_steps: int = 200,
    theta_unknown: float = 0.5,
    gate_enabled: bool = True,
    seed: int = 0,
) -> A3Result:
    """Simulate `n_steps` streaming batches; at each step, with probability
    `poison_rate`, attempt to inject a poison embedding for `class_idx`.

    When `gate_enabled`, poison is applied only if it passes the evidence
    gate; when disabled, poison is applied unconditionally (the no-defence
    control used to demonstrate what the gate is worth — docs §4).
    """
    torch.manual_seed(seed)
    bank = head.prototype_bank
    mask = torch.tensor([i == class_idx for i in bank.class_of])
    p0 = bank.bank.data[mask][0].clone()

    drift_curve: list[float] = []
    accepted, attempted = 0, 0
    with torch.no_grad():
        for step in range(n_steps):
            if torch.rand(1).item() > poison_rate:
                drift_curve.append(float(torch.norm(bank.bank.data[mask][0] - p0).item()))
                continue
            attempted += 1
            z = craft_poison_embedding(head, class_idx, seed=seed * 10_000 + step)
            ok = (not gate_enabled) or passes_evidence_gate(head, z, class_idx, theta_unknown)
            if ok:
                accepted += 1
                bank.ema_update(class_idx, z.unsqueeze(0), momentum)
            drift_curve.append(float(torch.norm(bank.bank.data[mask][0] - p0).item()))

    return A3Result(
        poison_rate=poison_rate,
        gate_enabled=gate_enabled,
        final_drift=drift_curve[-1] if drift_curve else 0.0,
        accepted_fraction=(accepted / attempted) if attempted else 0.0,
        drift_curve=drift_curve,
    )
