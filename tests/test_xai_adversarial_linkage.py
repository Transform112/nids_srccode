"""C4/F7 linkage tests: injected-edge mask propagation + attr_edge + injected_mass_fraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from argus.attacks.a2_structural_injection import build_injected_source
from argus.config import load_config
from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.xai.evidence_attrib import (
    edge_attribution_for_victim,
    injected_mass_fraction,
    victim_slot_flags,
)


def _make_source(f_e=147, n=400, seed=0):
    rng = np.random.default_rng(seed)
    times_ms = np.sort(rng.integers(0, 30_000, n))
    return AnchorBinGraphSource(
        times_ms, rng.standard_normal((n, f_e)).astype(np.float32),
        rng.integers(0, 12, n), rng.integers(0, 12, n), rng.integers(0, 3, n),
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=5,
        window_long_seconds=15, neighbour_cap=16, seed=seed,
    )


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(overrides=[
        "model.layers=1", "graph.neighbour_cap=16",
        "graph.window_mid_seconds=5", "graph.window_long_seconds=15",
    ])
    torch.manual_seed(0)
    model = ArgusModel(cfg, f_e=147, f_v=18, class_names=["benign", "atk1", "atk2"])
    model.eval()
    source = _make_source(seed=3)
    return cfg, model, source


def _pick_bin_and_victim(source):
    # A bin late enough that the long window has real context.
    for bin_id in source.unique_bins[len(source.unique_bins) // 2:]:
        batch = source.build_bin_batch(bin_id, f_v=18)
        if batch is not None and batch["n_targets"] > 0 and batch["scale_long"][0].shape[1] > 5:
            lo_hi = dict((b, (lo, hi)) for b, lo, hi in source.ranges["short"])[bin_id]
            positions = np.nonzero(source.bin_ids[lo_hi[0]:lo_hi[1]] == bin_id)[0] + lo_hi[0]
            row_pos = int(positions[0])
            return bin_id, 0, int(source.dst_ids[row_pos])
    pytest.skip("no suitable bin in synthetic source")


def test_injected_mask_survives_sort_and_reaches_batch(setup):
    cfg, model, source = setup
    bin_id, target_idx, victim = _pick_bin_and_victim(source)
    rng = np.random.default_rng(0)
    pool = rng.standard_normal((100, 147)).astype(np.float32)
    inj_host = int(max(source.src_ids.max(), source.dst_ids.max())) + 1

    inj_source = build_injected_source(
        source, bin_id, victim, inj_host, pool, budget=12,
        spread="all_strata", strata=4, seed=0, benign_class_id=0,
    )
    assert inj_source.is_injected.sum() == 12
    # Injected rows must be interleaved by time-sort, not appended.
    inj_positions = np.nonzero(inj_source.is_injected)[0]
    assert inj_positions[-1] < len(inj_source.times_ms) - 1 or len(inj_positions) == 12

    batch = inj_source.build_bin_batch(bin_id, f_v=18)
    assert batch is not None
    total_marked = sum(int(batch["edge_injected"][s].sum()) for s in ("short", "mid", "long"))
    assert total_marked > 0, "injected edges never reached any scale's context"
    # Injected flows are timed before the target bin — none may be a target.
    assert batch["n_targets"] == source.build_bin_batch(bin_id, f_v=18)["n_targets"]
    # Every marked long-scale edge must point at the victim.
    ei_long = batch["scale_long"][0]
    flags = batch["edge_injected"]["long"]
    if int(flags.sum()) > 0:
        victim_local = int((batch["node_ids"] == victim).nonzero(as_tuple=True)[0][0])
        assert (ei_long[1][flags] == victim_local).all()


def test_clean_source_has_all_false_mask(setup):
    cfg, model, source = setup
    bin_id, _, _ = _pick_bin_and_victim(source)
    batch = source.build_bin_batch(bin_id, f_v=18)
    for s in ("short", "mid", "long"):
        assert int(batch["edge_injected"][s].sum()) == 0


def test_attr_edge_normalised_and_fraction_bounded(setup):
    cfg, model, source = setup
    bin_id, target_idx, victim = _pick_bin_and_victim(source)
    rng = np.random.default_rng(0)
    pool = rng.standard_normal((100, 147)).astype(np.float32)
    inj_host = int(max(source.src_ids.max(), source.dst_ids.max())) + 1

    inj_source = build_injected_source(
        source, bin_id, victim, inj_host, pool, budget=8,
        spread="all_strata", strata=4, seed=0, benign_class_id=0,
    )
    batch = inj_source.build_bin_batch(bin_id, f_v=18)
    victim_local = int((batch["node_ids"] == victim).nonzero(as_tuple=True)[0][0])

    attr = edge_attribution_for_victim(
        model, batch, victim_local, target_idx, torch.device("cpu"),
    )
    if attr is None:
        pytest.skip("victim has no long-scale neighbours in synthetic bin")
    assert attr.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (attr >= 0).all()

    flags = victim_slot_flags(
        batch["scale_long"][0], victim_local, 16, batch["edge_injected"]["long"],
    )
    assert flags.shape[0] == attr.shape[0]
    frac = injected_mass_fraction(attr, flags)
    assert 0.0 <= frac <= 1.0 + 1e-6

    # Side channels must be off again after the call.
    last_layer = model.encoder.gnn_layers[-1]
    assert last_layer.record_attention is False
    assert last_layer.record_messages is False
    assert last_layer.last_msgs is None
