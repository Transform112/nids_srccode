"""End-to-end training loop test on synthetic data (Stage 1 + Stage 2)."""

from __future__ import annotations

import json

import numpy as np
import torch

from argus.config import load_config
from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.train.stage1_encoder import train_stage1
from argus.train.stage2_head import train_stage2


def _make_source(f_e=147, n=300, seed=0, label_p=None):
    rng = np.random.default_rng(seed)
    # Small time span keeps the number of anchor bins (and thus wall-clock time
    # of this unit test) low while still exercising multiple bins end-to-end.
    times_ms = np.sort(rng.integers(0, 20_000, n))
    src_ids = rng.integers(0, 15, n)
    dst_ids = rng.integers(0, 15, n)
    edge_features = rng.standard_normal((n, f_e)).astype(np.float32)
    labels = rng.choice(3, size=n, p=label_p) if label_p else rng.integers(0, 3, n)
    return AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=5,
        window_long_seconds=15, neighbour_cap=8, seed=seed,
    )


def test_stage1_and_stage2_training_run_end_to_end():
    cfg = load_config(overrides=[
        "model.layers=1",
        "graph.neighbour_cap=8",
        "graph.window_mid_seconds=5",
        "graph.window_long_seconds=15",
        "train.stage1_epochs=1",
        "train.stage2_epochs=1",
        "train.stage1_patience=1",
        "train.stage2_patience=1",
    ])
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    model = ArgusModel(cfg, f_e=147, f_v=18, class_names=class_names)
    device = torch.device("cpu")

    train_source = _make_source(seed=1)
    val_source = _make_source(seed=2)

    result1 = train_stage1(model, train_source, val_source, cfg, device)
    assert "history" in result1
    assert len(result1["history"]) >= 1

    result2 = train_stage2(model, train_source, val_source, cfg, device)
    assert "history" in result2
    assert len(result2["history"]) >= 1

    # Encoder must be frozen after Stage 2 setup.
    for p in model.encoder.parameters():
        assert not p.requires_grad


def test_stage1_wires_the_imbalance_mechanisms(tmp_path):
    """train_stage1 with `class_counts` must actually apply docs/06_TRAINING.md
    §4 end-to-end: weighted loss, class-balanced targets, min-count guard, and
    macro-F1 selection with a recorded G8 gate (docs/BUGS.md #50-#53)."""
    cfg = load_config(overrides=[
        "model.layers=1", "graph.neighbour_cap=8",
        "graph.window_mid_seconds=5", "graph.window_long_seconds=15",
        "train.stage1_epochs=1", "train.stage1_patience=1",
        "train.n_per_class=4", "train.min_count_for_prototype=100",
    ])
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    # Bot is the min-count class: 12 rows, below min_count_for_prototype=100.
    class_counts = {"Benign": 50_000, "FTP-BruteForce": 4_000, "Bot": 12}
    model = ArgusModel(cfg, f_e=147, f_v=18, class_names=class_names,
                       class_counts=class_counts)
    device = torch.device("cpu")

    result = train_stage1(
        model, _make_source(seed=1, label_p=[0.8, 0.19, 0.01]),
        _make_source(seed=2, label_p=[0.8, 0.19, 0.01]),
        cfg, device, run_dir=tmp_path, class_counts=class_counts,
    )

    assert result["excluded_classes"] == ["Bot"], "min-count guard did not fire"
    assert "best_val_macro_f1" in result
    # Selection ran on macro-F1, and the excluded class is absent from it.
    assert "Bot" not in result["best_val_per_class_f1"]
    assert "val_macro_f1" in result["history"][0]

    gates = json.loads((tmp_path / "gates_report.json").read_text(encoding="utf-8"))
    assert any(g["gate"] == "G8" for g in gates), "G8 was never evaluated"
