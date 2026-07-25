"""Degree-capped recency-stratified neighbour sampling.

See docs/04_GRAPH_CONSTRUCTION.md §4. This is both the efficiency mechanism and
the C2 robustness mechanism: pure "most recent K" is trivially attackable.
"""

from __future__ import annotations

import numpy as np


def sample_neighbours_recency_stratified(
    dst_ids: np.ndarray,
    k: int = 32,
    strata: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select at most `k` edge indices per destination node, recency-stratified.

    Args:
        dst_ids: [E] destination node id per edge, for edges already restricted
            to one window (assumed already sorted ascending by time).
        k: neighbour cap.
        strata: number of equal time sub-intervals (Q).
        rng: numpy Generator for reproducibility.
    Returns:
        Sorted array of selected edge indices (subset of range(len(dst_ids))).
    """
    rng = rng or np.random.default_rng(0)
    unique_dst = np.unique(dst_ids)
    per_stratum_quota = k // strata
    remainder = k - per_stratum_quota * strata

    selected: list[int] = []
    for v in unique_dst:
        edge_idx = np.nonzero(dst_ids == v)[0]
        if len(edge_idx) <= k:
            selected.extend(edge_idx.tolist())
            continue
        bin_edges = np.linspace(0, len(edge_idx), strata + 1, dtype=int)
        for i in range(strata):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bucket = edge_idx[lo:hi]
            quota = per_stratum_quota + (remainder if i == strata - 1 else 0)
            quota = min(quota, len(bucket))
            if quota > 0:
                chosen = rng.choice(bucket, size=quota, replace=False)
                selected.extend(chosen.tolist())

    return np.array(sorted(selected), dtype=np.int64)


def sample_neighbours_recent(dst_ids: np.ndarray, k: int = 32) -> np.ndarray:
    """Baseline: take the k most recent incident edges per node (attackable)."""
    selected: list[int] = []
    unique_dst = np.unique(dst_ids)
    for v in unique_dst:
        edge_idx = np.nonzero(dst_ids == v)[0]
        selected.extend(edge_idx[-k:].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def sample_neighbours_uniform(
    dst_ids: np.ndarray, k: int = 32, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Baseline: uniform sample without replacement per node."""
    rng = rng or np.random.default_rng(0)
    selected: list[int] = []
    unique_dst = np.unique(dst_ids)
    for v in unique_dst:
        edge_idx = np.nonzero(dst_ids == v)[0]
        if len(edge_idx) <= k:
            selected.extend(edge_idx.tolist())
        else:
            chosen = rng.choice(edge_idx, size=k, replace=False)
            selected.extend(chosen.tolist())
    return np.array(sorted(selected), dtype=np.int64)


def sample_neighbours(
    dst_ids: np.ndarray,
    k: int = 32,
    strategy: str = "recency_stratified",
    strata: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dispatch to the configured sampling strategy."""
    if strategy == "recency_stratified":
        return sample_neighbours_recency_stratified(dst_ids, k=k, strata=strata, rng=rng)
    if strategy == "recent":
        return sample_neighbours_recent(dst_ids, k=k)
    if strategy == "uniform":
        return sample_neighbours_uniform(dst_ids, k=k, rng=rng)
    raise ValueError(f"Unknown sampling strategy: {strategy}")
