"""Split protocols A (closed-set), B (open-set leave-classes-out), B2 (within-family).

See docs/02_DATASETS.md §4.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argus.constants import FAMILY_OF


def protocol_a_split(
    df: pd.DataFrame,
    label_col: str = "canonical_label",
    time_col: str = "FLOW_START_MILLISECONDS",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Per-class stratified temporal split. Every class appears in every split."""
    train_parts, val_parts, test_parts = [], [], []
    for _c, part in df.groupby(label_col, sort=False):
        part = part.sort_values(time_col)
        n = len(part)
        n_train = int(train_frac * n)
        n_val = int((train_frac + val_frac) * n)
        train_parts.append(part.iloc[:n_train])
        val_parts.append(part.iloc[n_train:n_val])
        test_parts.append(part.iloc[n_val:])

    def _finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
        out = pd.concat(parts, ignore_index=True)
        return out.sort_values(time_col).reset_index(drop=True)

    return {
        "train": _finalize(train_parts),
        "val": _finalize(val_parts),
        "test": _finalize(test_parts),
    }


def protocol_b_split(
    df: pd.DataFrame,
    holdout_classes: list[str],
    label_col: str = "canonical_label",
    time_col: str = "FLOW_START_MILLISECONDS",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    benign_label: str = "benign",
) -> dict[str, pd.DataFrame]:
    """Open-set split: hold out `holdout_classes`, labelled UNKNOWN at test time.

    Train/val exclude the holdout classes entirely. Test is the Protocol-A test
    split (restricted to known classes) plus all rows of the holdout classes.
    """
    known_df = df[~df[label_col].isin(holdout_classes)]
    unknown_df = df[df[label_col].isin(holdout_classes)]

    known_splits = protocol_a_split(known_df, label_col, time_col, train_frac, val_frac)
    unknown_labelled = unknown_df.copy()
    # Preserve the true class before relabelling: few-shot evaluation (T4)
    # needs per-held-out-class identity, which "UNKNOWN" erases.
    unknown_labelled["label_pre_holdout"] = unknown_labelled[label_col]
    unknown_labelled[label_col] = "UNKNOWN"

    known_test = known_splits["test"].copy()
    known_test["label_pre_holdout"] = known_test[label_col]
    test = pd.concat([known_test, unknown_labelled], ignore_index=True)
    test = test.sort_values(time_col).reset_index(drop=True)

    return {
        "train": known_splits["train"],
        "val": known_splits["val"],
        "test": test,
        "holdout_classes": holdout_classes,
    }


def protocol_b2_split(
    df: pd.DataFrame,
    held_out_variant: str,
    label_col: str = "canonical_label",
    time_col: str = "FLOW_START_MILLISECONDS",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Within-family open-set split (CICIDS2018 only).

    Hold out a single variant of a family while keeping its siblings in training.
    """
    if held_out_variant not in FAMILY_OF:
        raise ValueError(f"'{held_out_variant}' has no known family; cannot run Protocol B2")
    return protocol_b_split(
        df, [held_out_variant], label_col, time_col, train_frac, val_frac
    )


def sample_holdout_sets(
    attack_classes: list[str],
    holdout_size: int = 3,
    repeats: int = 5,
    seed: int = 1234,
) -> list[list[str]]:
    """Sample `repeats` distinct random holdout sets of size `holdout_size`."""
    rng = np.random.default_rng(seed)
    holdouts = []
    seen = set()
    attempts = 0
    while len(holdouts) < repeats and attempts < repeats * 50:
        attempts += 1
        choice = tuple(sorted(rng.choice(attack_classes, size=holdout_size, replace=False)))
        if choice not in seen:
            seen.add(choice)
            holdouts.append(list(choice))
    return holdouts
