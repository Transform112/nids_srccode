"""Graph construction tests: windows, sampler, batching invariants."""

from __future__ import annotations

import numpy as np
import pytest

from argus.graph.batching import AnchorBinGraphSource
from argus.graph.sampler import sample_neighbours_recency_stratified
from argus.graph.windows import assign_anchor_bins, window_edge_ranges


def _synthetic_flows(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    times_ms = np.sort(rng.integers(0, 600_000, n))  # 600s span
    src_ids = rng.integers(0, 50, n)
    dst_ids = rng.integers(0, 50, n)
    edge_features = rng.standard_normal((n, 12)).astype(np.float32)
    labels = rng.integers(0, 3, n)
    return times_ms, edge_features, src_ids, dst_ids, labels


def test_no_lookahead_in_window_ranges():
    times_ms, *_ = _synthetic_flows()
    bin_ids = assign_anchor_bins(times_ms, anchor_bin_seconds=1)
    ranges = window_edge_ranges(times_ms, bin_ids, scale_seconds=30)
    for b, lo, hi in ranges:
        if hi == 0:
            continue
        window_end = times_ms[hi - 1]
        assert (times_ms[lo:hi] <= window_end).all()


def test_sample_neighbours_never_exceeds_k():
    rng = np.random.default_rng(0)
    dst_ids = rng.integers(0, 5, 500)
    selected = sample_neighbours_recency_stratified(dst_ids, k=8, strata=4, rng=rng)
    for v in np.unique(dst_ids):
        count = np.isin(selected, np.nonzero(dst_ids == v)[0]).sum()
        assert count <= 8


def test_sample_neighbours_stratified_bounds_share():
    """No single time stratum should exceed ceil(K/Q) edges for a well-populated node."""
    dst_ids = np.zeros(200, dtype=np.int64)  # all edges to node 0
    rng = np.random.default_rng(1)
    selected = sample_neighbours_recency_stratified(dst_ids, k=32, strata=4, rng=rng)
    assert len(selected) <= 32
    # Check quarter distribution: split selected indices into 4 chronological buckets.
    bin_edges = np.linspace(0, 200, 5, dtype=int)
    for i in range(4):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        count_in_bucket = ((selected >= lo) & (selected < hi)).sum()
        assert count_in_bucket <= 32 // 4 + 1


def test_anchor_bin_graph_source_target_in_short_scale():
    times_ms, edge_features, src_ids, dst_ids, labels = _synthetic_flows()
    source = AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=30,
        window_long_seconds=300, neighbour_cap=8,
    )
    found_any = False
    for b in source.unique_bins[:20]:
        batch = source.build_bin_batch(b, f_v=4)
        if batch is None or batch["n_targets"] == 0:
            continue
        found_any = True
        assert batch["target_edge_index"].shape[1] == batch["n_targets"]
        assert batch["target_edge_attr"].shape[0] == batch["n_targets"]
        max_node_idx = batch["target_edge_index"].max().item() if batch["target_edge_index"].numel() else -1
        assert max_node_idx < batch["node_feat"].shape[0]
    assert found_any


def test_reproducible_given_fixed_seed():
    times_ms, edge_features, src_ids, dst_ids, labels = _synthetic_flows()
    kwargs = dict(
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=30,
        window_long_seconds=300, neighbour_cap=8, seed=42,
    )
    s1 = AnchorBinGraphSource(times_ms, edge_features, src_ids, dst_ids, labels, **kwargs)
    s2 = AnchorBinGraphSource(times_ms, edge_features, src_ids, dst_ids, labels, **kwargs)
    b = s1.unique_bins[10]
    batch1 = s1.build_bin_batch(b, f_v=4)
    batch2 = s2.build_bin_batch(b, f_v=4)
    if batch1 is None or batch2 is None:
        pytest.skip("bin has no data")
    assert torch_equal_or_both_empty(batch1["scale_short"][0], batch2["scale_short"][0])


def test_node_features_are_real_signal_not_zeros():
    """Node features must be computed from real graph statistics (docs/04 §3),
    not left as a zero placeholder."""
    times_ms, edge_features, src_ids, dst_ids, labels = _synthetic_flows(n=3000)
    source = AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=30,
        window_long_seconds=300, neighbour_cap=8,
    )
    found_nonzero = False
    for b in source.unique_bins[:50]:
        batch = source.build_bin_batch(b, f_v=18)
        if batch is None:
            continue
        if batch["node_feat"].abs().sum().item() > 0:
            found_nonzero = True
            break
    assert found_nonzero, "node_feat is all zeros across sampled bins"


def torch_equal_or_both_empty(a, b):
    import torch
    if a.numel() == 0 and b.numel() == 0:
        return True
    return torch.equal(a, b)
