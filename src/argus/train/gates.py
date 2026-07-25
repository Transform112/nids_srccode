"""Sanity gates G0-G7. See docs/06_TRAINING.md §8.

Each gate function returns (passed: bool, message: str).
"""

from __future__ import annotations

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
