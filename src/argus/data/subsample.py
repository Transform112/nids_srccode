"""Stratified temporal subsampling.

See docs/02_DATASETS.md §5.3. Never uniform random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_temporal_subsample(
    df: pd.DataFrame,
    target_total: int,
    minority_threshold: int = 200_000,
    benign_floor_fraction: float = 0.50,
    benign_label: str = "benign",
    time_bins: int = 100,
    label_col: str = "canonical_label",
    time_col: str = "FLOW_START_MILLISECONDS",
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Subsample a dataframe to `target_total` rows via per-class temporal strata.

    Rule (docs/02_DATASETS.md §5.3):
        - Classes at or below `minority_threshold` are kept in full.
        - Larger classes are split into `time_bins` consecutive temporal bins and
          sampled proportionally from each bin, without replacement.
        - Benign quota is floored at `benign_floor_fraction` of the total budget.

    Returns:
        (subsampled_df, report dict with pre/post class histograms)
    """
    rng = np.random.default_rng(seed)
    counts = df[label_col].value_counts()
    report: dict = {"pre_counts": counts.to_dict()}

    kept_frames = []
    minority_classes = counts[counts <= minority_threshold].index.tolist()
    majority_classes = [c for c in counts.index if c not in minority_classes]

    # Keep all minority-class rows.
    minority_total = 0
    for c in minority_classes:
        part = df[df[label_col] == c]
        kept_frames.append(part)
        minority_total += len(part)

    remaining_budget = max(target_total - minority_total, 0)

    # Benign floor.
    benign_floor = int(benign_floor_fraction * target_total)
    non_benign_majority = [c for c in majority_classes if c != benign_label]

    quotas: dict[str, int] = {}
    if benign_label in majority_classes:
        quotas[benign_label] = min(counts[benign_label], max(benign_floor, 0))
        remaining_after_benign = max(remaining_budget - quotas[benign_label], 0)
    else:
        remaining_after_benign = remaining_budget

    if non_benign_majority:
        per_class_share = remaining_after_benign // max(len(non_benign_majority), 1)
        for c in non_benign_majority:
            quotas[c] = min(counts[c], per_class_share)

    # If benign wasn't in majority classes but is a minority class, it's already kept fully.

    for c, quota in quotas.items():
        part = df[df[label_col] == c].sort_values(time_col)
        n = len(part)
        if quota >= n:
            kept_frames.append(part)
            continue
        bin_edges = np.linspace(0, n, time_bins + 1, dtype=int)
        per_bin_quota = quota // time_bins
        remainder = quota - per_bin_quota * time_bins
        selected_idx = []
        for i in range(time_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_size = hi - lo
            if bin_size <= 0:
                continue
            take = per_bin_quota + (1 if i < remainder else 0)
            take = min(take, bin_size)
            if take > 0:
                chosen = rng.choice(bin_size, size=take, replace=False) + lo
                selected_idx.extend(chosen.tolist())
        kept_frames.append(part.iloc[sorted(selected_idx)])

    result = pd.concat(kept_frames, ignore_index=True)
    result = result.sort_values(time_col).reset_index(drop=True)
    report["post_counts"] = result[label_col].value_counts().to_dict()
    report["target_total"] = target_total
    report["realised_total"] = len(result)
    return result, report
