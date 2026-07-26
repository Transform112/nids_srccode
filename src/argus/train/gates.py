"""Sanity gates G0-G7. See docs/06_TRAINING.md §8.

Each gate function returns (passed: bool, message: str).

`record_gate` appends results to a per-run `gates_report.json`;
`prototype_gate_stats` computes the geometry inputs for G2/G2b from a
prototype bank.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def gate_g0_capacity(train_acc: float, required: float = 0.99) -> tuple[bool, str]:
    passed = train_acc >= required
    return passed, f"G0 capacity check: train_acc={train_acc:.4f} (required >= {required})"


def gate_g1_encoder_learning(val_f1_epoch5: float, min_f1: float = 0.5) -> tuple[bool, str]:
    passed = val_f1_epoch5 >= min_f1
    return passed, f"G1 encoder learning: val_f1={val_f1_epoch5:.4f} (required >= {min_f1})"


def gate_g2_prototype_collapse(mean_inter_class_cosine: float, max_cosine: float = 0.8) -> tuple[bool, str]:
    passed = mean_inter_class_cosine <= max_cosine
    return passed, f"G2 prototype geometry: inter-class cos={mean_inter_class_cosine:.4f} (max {max_cosine})"


def gate_g2b_subprototype_collapse(mean_intra_class_cosine: float, max_cosine: float = 0.8) -> tuple[bool, str]:
    passed = mean_intra_class_cosine <= max_cosine
    return passed, f"G2b sub-prototype diversity: intra-class cos={mean_intra_class_cosine:.4f} (max {max_cosine})"


def gate_g3_evidence_collapse(mean_known_vacuity: float, max_vacuity: float = 0.7) -> tuple[bool, str]:
    passed = mean_known_vacuity <= max_vacuity
    return passed, f"G3 evidence collapse: mean known vacuity={mean_known_vacuity:.4f} (max {max_vacuity})"


def gate_g4_unknown_carving(mean_unknown_vacuity: float, min_vacuity: float = 0.5) -> tuple[bool, str]:
    passed = mean_unknown_vacuity >= min_vacuity
    return passed, f"G4 unknown carving: mean synthetic-unknown vacuity={mean_unknown_vacuity:.4f} (min {min_vacuity})"


def gate_g5_channel_reliance(channel_ratio: float, max_ratio: float = 0.8) -> tuple[bool, str]:
    passed = channel_ratio <= max_ratio
    return passed, f"G5 channel reliance: ratio={channel_ratio:.4f} (max {max_ratio})"


def gate_g6_numerical_health(loss: torch.Tensor) -> tuple[bool, str]:
    passed = bool(torch.isfinite(loss).all())
    return passed, f"G6 numerical health: loss finite={passed} (value={loss.item() if passed else 'NaN/Inf'})"


def gate_g7_overfitting(train_f1: float, val_f1: float, max_gap: float = 0.10) -> tuple[bool, str]:
    gap = train_f1 - val_f1
    passed = gap <= max_gap
    return passed, f"G7 overfitting: train-val gap={gap:.4f} (max {max_gap})"


def gate_g8_tail_collapse(
    per_class_f1: dict[str, float],
    max_collapsed: int = 0,
    min_f1: float = 0.0,
) -> tuple[bool, str]:
    """Fail when any trained class scores at or below `min_f1` on validation.

    G0 cannot catch this: it trains on a *class-balanced* subset, so a model
    that has stopped predicting the tail entirely still memorises that subset
    and passes at 0.99. Three consecutive 12-hour runs shipped a near-constant
    single-class predictor with every capacity gate green (docs/BUGS.md #49).

    Aggregate metrics cannot catch it either — on this split, ignoring the six
    rarest classes costs 0.2 points of accuracy. Only a per-class floor makes
    the failure visible, which is what this gate is.

    `per_class_f1` should exclude minimum-count classes: those are deliberately
    not trained in Stage 1 (docs/06_TRAINING.md §4.3), so scoring them here
    would fail the gate by design.
    """
    collapsed = sorted(name for name, f1 in per_class_f1.items() if f1 <= min_f1)
    passed = len(collapsed) <= max_collapsed
    detail = f" [{', '.join(collapsed)}]" if collapsed else ""
    return passed, (
        f"G8 tail collapse: {len(collapsed)}/{len(per_class_f1)} trained classes "
        f"at F1<={min_f1:g}{detail} (max {max_collapsed})"
    )


def record_gate(report_path: str | Path | None, name: str, passed: bool, message: str) -> None:
    """Print a gate result and append it to `gates_report.json` (if a path is given)."""
    status = "PASS" if passed else "FAIL"
    print(f"[gate] {status}  {message}", flush=True)
    if report_path is None:
        return
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if report_path.is_file():
        try:
            entries = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
    entries.append({
        "gate": name,
        "passed": passed,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    report_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def prototype_gate_stats(prototype_bank) -> tuple[float, float]:
    """Compute (mean inter-class centroid cosine, mean intra-class sub-prototype cosine).

    Inputs for G2/G2b. Intra-class mean is 0.0 when no class has more than one
    sub-prototype (nothing to collapse).
    """
    import torch.nn.functional as F

    bank = F.normalize(prototype_bank.bank.detach(), dim=-1)
    class_of = torch.tensor(prototype_bank.class_of)
    classes = sorted(set(prototype_bank.class_of))

    centroids = torch.stack([
        F.normalize(bank[class_of == c].mean(dim=0), dim=-1) for c in classes
    ])
    sim = centroids @ centroids.T
    n = len(classes)
    if n > 1:
        triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        inter = float(sim[triu].mean())
    else:
        inter = 0.0

    intra_vals = []
    for c in classes:
        sub = bank[class_of == c]
        if sub.shape[0] > 1:
            s = sub @ sub.T
            m = sub.shape[0]
            triu = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)
            intra_vals.append(float(s[triu].mean()))
    intra = float(sum(intra_vals) / len(intra_vals)) if intra_vals else 0.0
    return inter, intra
