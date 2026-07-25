"""Robust aggregation tests."""

import pytest
import torch

from argus.models.aggregation import (
    RobustAggregator,
    soft_medoid,
    trimmed_mean,
    trimmed_std,
)


def test_trimmed_mean_breakdown_point():
    """Corrupted values that fall into the trimmed tails do not affect the aggregate."""
    torch.manual_seed(0)
    msgs = torch.randn(20, 8)
    weights = torch.ones(20) / 20.0
    beta = 0.20
    k = int(beta * 20)  # 4

    # Force corruption into the tails so trimming removes it exactly.
    corrupted = msgs.clone()
    order = msgs.argsort(dim=0)
    low_tail = order[:k]
    high_tail = order[-k:]
    for d in range(msgs.shape[1]):
        corrupted[low_tail[:, d], d] = -1e6
        corrupted[high_tail[:, d], d] = 1e6
    clean = trimmed_mean(msgs, weights, beta=beta)
    perturbed = trimmed_mean(corrupted, weights, beta=beta)
    assert torch.allclose(clean, perturbed, atol=1e-4)


def test_trimmed_mean_moves_when_majority_corrupted():
    torch.manual_seed(0)
    msgs = torch.randn(20, 8)
    weights = torch.ones(20) / 20.0
    beta = 0.20
    clean = trimmed_mean(msgs, weights, beta=beta)

    corrupted = msgs.clone()
    corrupted[:12] += 50.0  # > beta*n from each tail
    perturbed = trimmed_mean(corrupted, weights, beta=beta)
    assert not torch.allclose(clean, perturbed, atol=1e-2)


def test_trimmed_std_nonnegative():
    msgs = torch.randn(10, 4)
    weights = torch.ones(10) / 10.0
    std = trimmed_std(msgs, weights, beta=0.2)
    assert (std >= 0).all()


def test_soft_medoid_robust():
    torch.manual_seed(0)
    msgs = torch.randn(16, 4)
    weights = torch.ones(16) / 16.0
    clean = soft_medoid(msgs, weights, temperature=1.0)

    corrupted = msgs.clone()
    corrupted[0] += 100.0  # single outlier
    perturbed = soft_medoid(corrupted, weights, temperature=1.0)
    # Medoid should be far less affected than mean
    mean_clean = (msgs * weights.unsqueeze(-1)).sum(0)
    mean_corrupt = (corrupted * weights.unsqueeze(-1)).sum(0)
    delta_medoid = (clean - perturbed).norm()
    delta_mean = (mean_clean - mean_corrupt).norm()
    assert delta_medoid < delta_mean


def test_robust_aggregator_dispatch():
    agg = RobustAggregator(d_h=8, aggregation="trimmed", beta=0.2, k=8)
    msgs = torch.randn(5, 8)
    weights = torch.ones(5) / 5.0
    out = agg(msgs, weights)
    assert out.shape == (8,)


def test_mean_aggregator_matches_plain():
    agg = RobustAggregator(d_h=4, aggregation="mean", multi_aggregator=False)
    msgs = torch.randn(6, 4)
    weights = torch.ones(6) / 6.0
    out = agg(msgs, weights)
    assert torch.allclose(out, (msgs * weights.unsqueeze(-1)).sum(0), atol=1e-6)
