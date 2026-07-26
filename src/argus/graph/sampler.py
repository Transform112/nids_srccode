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
    age_seconds: np.ndarray | None = None,
    window_seconds: float | None = None,
) -> np.ndarray:
    """Select at most `k` edge indices per destination node, recency-stratified.

    Args:
        dst_ids: [E] destination node id per edge, for edges already restricted
            to one window (assumed already sorted ascending by time).
        k: neighbour cap.
        strata: number of equal sub-intervals (Q) of the window.
        rng: numpy Generator for reproducibility.
        age_seconds: [E] seconds before the window end (0 = most recent).
            Required for genuine *time*-based stratification
            (docs/04_GRAPH_CONSTRUCTION.md §4: "split the window into 4 equal
            sub-intervals"). Without it, falls back to splitting by array
            position — equal-*count*, not equal-*time* — which lets a burst
            large enough to span multiple count-quantiles claim more than
            1/Q of the sample despite arriving within a single narrow
            instant, weakening the anti-burst-injection guarantee C2 relies
            on.
        window_seconds: the scale's declared window duration (e.g.
            `window_long_seconds`). Sub-interval boundaries are fixed
            fractions of *this* — an attacker-independent constant — not of
            whatever time span the currently-present edges happen to span,
            so an adversary can't shift bucket boundaries by controlling how
            sparse or dense the surrounding traffic is.
    Returns:
        Sorted array of selected edge indices (subset of range(len(dst_ids))).
    """
    rng = rng or np.random.default_rng(0)
    unique_dst = np.unique(dst_ids)
    per_stratum_quota = k // strata
    remainder = k - per_stratum_quota * strata

    use_time = age_seconds is not None and window_seconds is not None and window_seconds > 0
    if use_time:
        bin_width = window_seconds / strata
        # bucket 0 = most recent (age in [0, bin_width)) ... bucket Q-1 = oldest
        edge_bucket = np.clip((age_seconds / bin_width).astype(np.int64), 0, strata - 1)

    selected: list[int] = []
    for v in unique_dst:
        edge_idx = np.nonzero(dst_ids == v)[0]
        if len(edge_idx) <= k:
            selected.extend(edge_idx.tolist())
            continue

        if use_time:
            buckets = [edge_idx[edge_bucket[edge_idx] == s] for s in range(strata)]
        else:
            bin_edges = np.linspace(0, len(edge_idx), strata + 1, dtype=int)
            buckets = [edge_idx[bin_edges[i]:bin_edges[i + 1]] for i in range(strata)]

        quotas = [per_stratum_quota] * strata
        quotas[0] += remainder  # k % strata also goes to the most-recent stratum

        achievable = [min(quotas[s], len(buckets[s])) for s in range(strata)]
        shortfall = sum(quotas[s] - achievable[s] for s in range(strata))
        if shortfall > 0:
            # Redistribute short strata's unfilled quota to the most recent one.
            achievable[0] = min(achievable[0] + shortfall, len(buckets[0]))

        for s in range(strata):
            bucket, quota = buckets[s], achievable[s]
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
    age_seconds: np.ndarray | None = None,
    window_seconds: float | None = None,
) -> np.ndarray:
    """Dispatch to the configured sampling strategy."""
    if strategy == "recency_stratified":
        return sample_neighbours_recency_stratified(
            dst_ids, k=k, strata=strata, rng=rng,
            age_seconds=age_seconds, window_seconds=window_seconds,
        )
    if strategy == "recent":
        return sample_neighbours_recent(dst_ids, k=k)
    if strategy == "uniform":
        return sample_neighbours_uniform(dst_ids, k=k, rng=rng)
    raise ValueError(f"Unknown sampling strategy: {strategy}")
