"""Threshold calibration tests (validation-only theta selection)."""

from __future__ import annotations

import numpy as np
import torch

from argus.config import load_config
from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.train.thresholds import calibrate_thresholds, select_theta_defer, select_theta_unknown


def _make_source(f_e=147, n=300, seed=0):
    rng = np.random.default_rng(seed)
    times_ms = np.sort(rng.integers(0, 20_000, n))
    src_ids = rng.integers(0, 15, n)
    dst_ids = rng.integers(0, 15, n)
    edge_features = rng.standard_normal((n, f_e)).astype(np.float32)
    labels = rng.integers(0, 3, n)
    return AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=5,
        window_long_seconds=15, neighbour_cap=8, seed=seed,
    )


def test_select_theta_unknown_quantile():
    evidence = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 10.0, 20.0, 30.0, 40.0, 50.0])
    theta = select_theta_unknown(evidence, target_false_unknown_rate=0.1)
    assert theta <= evidence.min().item() + 1e-6 or theta <= np.quantile(evidence.numpy(), 0.1) + 1e-6


def test_select_theta_defer_quantile():
    margins = torch.linspace(0, 1, 100)
    theta = select_theta_defer(margins, target_defer_rate=0.05)
    assert 0.0 <= theta <= 0.1


def test_calibrate_thresholds_end_to_end():
    cfg = load_config(overrides=["model.layers=1", "graph.neighbour_cap=8"])
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    model = ArgusModel(cfg, f_e=147, f_v=18, class_names=class_names)
    val_source = _make_source(seed=2)
    result = calibrate_thresholds(model, val_source, torch.device("cpu"), max_bins=10)
    assert "theta_unknown" in result and "theta_defer" in result
    assert model.head.theta_unknown == result["theta_unknown"]
    assert model.head.theta_defer == result["theta_defer"]
