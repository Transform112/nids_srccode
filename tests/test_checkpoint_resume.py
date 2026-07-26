"""Kill-and-resume trajectory test (docs/06_TRAINING.md §6.2).

Trains 2 epochs, "kills" the session, resumes to 4 epochs, and compares
against an uninterrupted 4-epoch run. Bit-identical is unrealistic with AMP
on GPU; on CPU with full RNG restore the trajectories should agree to well
within the doc's 1e-3 tolerance.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from argus.config import load_config
from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.train.stage1_encoder import train_stage1


def _make_source(f_e=147, n=300, seed=0):
    rng = np.random.default_rng(seed)
    times_ms = np.sort(rng.integers(0, 20_000, n))
    src_ids = rng.integers(0, 15, n)
    dst_ids = rng.integers(0, 15, n)
    edge_features = rng.standard_normal((n, f_e)).astype(np.float32)
    labels = rng.integers(0, 3, n)
    # neighbour_cap high enough that no stochastic sampling occurs — the graph
    # source's own RNG is not checkpointed, so determinism must come from the
    # data itself here.
    return AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=5,
        window_long_seconds=15, neighbour_cap=512, seed=seed,
    )


def _cfg(epochs: int):
    return load_config(overrides=[
        "model.layers=1",
        "graph.neighbour_cap=512",
        "graph.window_mid_seconds=5",
        "graph.window_long_seconds=15",
        f"train.stage1_epochs={epochs}",
        "train.stage1_patience=10",
    ])


def _fresh_model(cfg, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ArgusModel(cfg, f_e=147, f_v=18, class_names=["a", "b", "c"])


def test_resume_reproduces_uninterrupted_trajectory(tmp_path):
    device = torch.device("cpu")

    # Sources are rebuilt fresh per run, exactly as a real killed-and-restarted
    # Kaggle session would rebuild them from disk.
    cfg4 = _cfg(4)
    model_a = _fresh_model(cfg4)
    result_a = train_stage1(model_a, _make_source(seed=1), _make_source(seed=2), cfg4, device,
                            run_dir=tmp_path / "run_a")

    # Interrupted: 2 epochs, then resume to 4 with a fresh process-equivalent
    # (new model object; checkpoint restores weights/optimizer/RNG).
    cfg2 = _cfg(2)
    model_b = _fresh_model(cfg2)
    train_stage1(model_b, _make_source(seed=1), _make_source(seed=2), cfg2, device,
                 run_dir=tmp_path / "run_b")

    cfg4b = _cfg(4)
    model_b2 = _fresh_model(cfg4b, seed=999)  # different init — must be overwritten by resume
    result_b = train_stage1(model_b2, _make_source(seed=1), _make_source(seed=2), cfg4b, device,
                            run_dir=tmp_path / "run_b", resume=True)

    hist_a = result_a["history"]
    hist_b = result_b["history"]
    assert len(hist_a) == 4
    assert len(hist_b) == 4, "resume must continue the same history, not restart it"
    assert [h["epoch"] for h in hist_b] == [0, 1, 2, 3]

    for ea, eb in zip(hist_a[2:], hist_b[2:]):
        assert ea["train_loss"] == pytest.approx(eb["train_loss"], abs=1e-3)
        assert ea["val_loss"] == pytest.approx(eb["val_loss"], abs=1e-3)


def test_best_weights_restored_on_return(tmp_path):
    device = torch.device("cpu")
    train_source = _make_source(seed=1)
    val_source = _make_source(seed=2)
    cfg = _cfg(3)
    model = _fresh_model(cfg)
    result = train_stage1(model, train_source, val_source, cfg, device, run_dir=tmp_path)

    best_ckpt = tmp_path / "stage1_ckpt_best.pt"
    assert best_ckpt.exists()
    state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    for k, v in model.state_dict().items():
        assert torch.equal(v, state["model_state"][k]), f"{k} differs from best checkpoint"
    assert result["best_epoch"] == state["extra"]["best_epoch"]


def test_optimizer_state_is_real_in_last_checkpoint(tmp_path):
    device = torch.device("cpu")
    train_source = _make_source(seed=1)
    val_source = _make_source(seed=2)
    cfg = _cfg(2)
    model = _fresh_model(cfg)
    result = train_stage1(model, train_source, val_source, cfg, device, run_dir=tmp_path)

    # The returned optimizer must be the live, stepped one.
    opt_state = result["optimizer"].state_dict()["state"]
    assert len(opt_state) > 0, "returned optimizer was never stepped"
    assert any("exp_avg" in s for s in opt_state.values())

    ckpt = torch.load(tmp_path / "stage1_ckpt_last.pt", map_location="cpu", weights_only=False)
    assert len(ckpt["optimizer_state"]["state"]) > 0, "checkpointed optimizer state is empty/dummy"
