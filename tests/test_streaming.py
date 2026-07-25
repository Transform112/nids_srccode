"""Streaming detector tests: no lookahead, bounded memory, register_class contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from argus.config import load_config
from argus.models.argus import ArgusModel
from argus.streaming.detector import StreamingDetector


def _make_detector(f_e=147):
    cfg = load_config(overrides=[
        "model.layers=1", "graph.neighbour_cap=8",
        "graph.window_mid_seconds=5", "graph.window_long_seconds=15",
    ])
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    model = ArgusModel(cfg, f_e=f_e, f_v=18, class_names=class_names)
    detector = StreamingDetector(
        model, class_names,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=5,
        window_long_seconds=15, neighbour_cap=8,
    )
    return detector


def test_push_returns_one_verdict_per_flow():
    detector = _make_detector()
    rng = np.random.default_rng(0)
    n = 10
    feats = rng.standard_normal((n, 147)).astype(np.float32)
    times = np.arange(1000, 1000 + n * 100, 100).astype(np.int64)
    src = rng.integers(0, 5, n)
    dst = rng.integers(0, 5, n)
    verdicts = detector.push(feats, times, src, dst)
    assert len(verdicts) == n
    for v in verdicts:
        assert v.decision in ("CLASSIFY", "DEFER", "UNKNOWN")
        assert v.latency_ms >= 0


def test_push_rejects_lookahead_violation():
    detector = _make_detector()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((5, 147)).astype(np.float32)
    times1 = np.array([5000, 5100, 5200, 5300, 5400], dtype=np.int64)
    src = rng.integers(0, 5, 5)
    dst = rng.integers(0, 5, 5)
    detector.push(feats, times1, src, dst)

    times2 = np.array([1000, 1100, 1200, 1300, 1400], dtype=np.int64)  # earlier than already pushed
    with pytest.raises(ValueError, match="lookahead"):
        detector.push(feats, times2, src, dst)


def test_memory_bounded_by_eviction():
    detector = _make_detector()
    rng = np.random.default_rng(0)
    for batch_idx in range(5):
        feats = rng.standard_normal((20, 147)).astype(np.float32)
        base_t = batch_idx * 20_000  # jump well beyond the 15s long window each time
        times = np.arange(base_t, base_t + 20 * 100, 100).astype(np.int64)
        src = rng.integers(0, 5, 20)
        dst = rng.integers(0, 5, 20)
        detector.push(feats, times, src, dst)
    # After several far-apart pushes, the buffer must not contain all 100 flows.
    assert detector.memory_footprint() < 100


def test_register_class_zero_parameter_change():
    detector = _make_detector()
    model = detector.model
    state_before = {k: v.clone() for k, v in model.head.state_dict().items()}

    rng = np.random.default_rng(0)
    new_samples = rng.standard_normal((5, 147)).astype(np.float32)
    n_classes_before = model.head.prototype_bank.num_classes
    detector.register_class("NewAttack", new_samples, n_sub=1)
    n_classes_after = model.head.prototype_bank.num_classes

    assert n_classes_after == n_classes_before + 1
    state_after = model.head.state_dict()
    for key, before in state_before.items():
        after = state_after[key]
        if key.endswith("bank"):
            assert torch.equal(before, after[: before.shape[0]])
        else:
            assert torch.equal(before, after)


def test_measure_streaming_throughput_reports_finite_percentiles():
    from argus.eval.deployment import measure_streaming_throughput, model_size

    detector = _make_detector()
    n_params, size_mb = model_size(detector.model)
    assert n_params > 0
    assert size_mb > 0

    rng = np.random.default_rng(0)
    n = 20
    feats = rng.standard_normal((n, 147)).astype(np.float32)
    times = np.arange(1000, 1000 + n * 100, 100).astype(np.int64)
    src = rng.integers(0, 5, n)
    dst = rng.integers(0, 5, n)

    report = measure_streaming_throughput(detector, feats, times, src, dst, batch_size=1)
    assert report.n_flows == n
    assert report.throughput_flows_per_sec > 0
    assert report.latency_p50_ms >= 0
    assert report.latency_p99_ms >= report.latency_p50_ms
    assert report.peak_memory_mb >= 0
    assert report.model_size_params == n_params
