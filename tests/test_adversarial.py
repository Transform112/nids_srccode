"""Adversarial A1-A5 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from argus.attacks.a1_feature_pgd import run_a1_epsilon_sweep
from argus.attacks.a2_structural_injection import (
    build_injected_source,
    run_a2_budget_sweep,
    sample_benign_like_flows,
)
from argus.attacks.a3_prototype_poison import craft_poison_embedding, run_a3_poison_sweep
from argus.attacks.a4_adaptive import run_a4_adaptive
from argus.attacks.a5_temporal_jitter import run_a5_jitter_sweep
from argus.attacks.constraints import assert_feasible, project
from argus.config import load_config
from argus.features.pipeline import FeaturePipeline
from argus.graph.batching import AnchorBinGraphSource
from argus.models.argus import ArgusModel
from argus.models.epc import EPCHead


def test_project_enforces_byte_floor():
    df = pd.DataFrame({
        "IN_PKTS": [10, 5],
        "IN_BYTES": [1, 1000],  # first row violates the header floor
        "OUT_PKTS": [2, 3],
        "OUT_BYTES": [1, 200],
    })
    projected = project(df)
    assert_feasible(projected)  # must not raise


def test_project_enforces_iat_ordering():
    df = pd.DataFrame({
        "SRC_TO_DST_IAT_MIN": [50.0],
        "SRC_TO_DST_IAT_AVG": [10.0],  # avg below min -> infeasible before projection
        "SRC_TO_DST_IAT_MAX": [30.0],  # max below avg -> infeasible before projection
        "SRC_TO_DST_IAT_STDDEV": [-5.0],
    })
    projected = project(df)
    assert_feasible(projected)


def test_sample_benign_like_flows_shapes():
    rng = np.random.default_rng(0)
    pool = rng.standard_normal((50, 12))
    sampled = sample_benign_like_flows(pool, m=10, rng=rng)
    assert sampled.shape == (10, 12)


def _make_source(f_e=12, n=500, seed=0):
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


def test_build_injected_source_adds_flows_without_new_targets():
    source = _make_source()
    bin_id = source.unique_bins[10]
    batch_before = source.build_bin_batch(bin_id, f_v=4)
    if batch_before is None:
        return
    rng = np.random.default_rng(1)
    benign_pool = rng.standard_normal((20, 12)).astype(np.float32)
    injected = build_injected_source(
        source, bin_id, victim_node_id=int(source.dst_ids[0]),
        injection_host_id=9999, benign_pool=benign_pool, budget=8,
        spread="all_strata",
    )
    batch_after = injected.build_bin_batch(bin_id, f_v=4)
    assert batch_after is not None
    # Injected flows must not become new targets of the attacked bin.
    assert batch_after["n_targets"] == batch_before["n_targets"]


def test_a2_budget_sweep_runs_end_to_end():
    cfg = load_config(overrides=["model.layers=1", "graph.neighbour_cap=8"])
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    model = ArgusModel(cfg, f_e=147, f_v=18, class_names=class_names)
    source = _make_source(f_e=147, seed=2)

    bin_id = None
    for b in source.unique_bins:
        batch = source.build_bin_batch(b, f_v=18)
        if batch is not None and batch["n_targets"] > 0:
            bin_id = b
            break
    assert bin_id is not None

    rng = np.random.default_rng(3)
    benign_pool = rng.standard_normal((30, 147)).astype(np.float32)
    results = run_a2_budget_sweep(
        model, source, bin_id, target_index_within_bin=0,
        victim_node_id=int(source.dst_ids[0]), injection_host_id=8888,
        benign_pool=benign_pool, device=torch.device("cpu"),
        budgets=[0, 2, 8], spread="all_strata",
    )
    assert len(results) == 3
    assert results[0].budget == 0
    assert results[0].clean_decision == results[0].attacked_decision
    for r in results:
        assert np.isfinite(r.clean_evidence_total)
        assert np.isfinite(r.attacked_evidence_total)


def _make_synthetic_raw_flows(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """NF-v3-shaped synthetic raw (pre-pipeline) flows, sorted by start time so
    row order matches `AnchorBinGraphSource`'s internal stable time-sort.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "FLOW_START_MILLISECONDS": np.sort(rng.integers(1_600_000_000_000, 1_600_100_000_000, n)),
        "FLOW_END_MILLISECONDS": np.sort(rng.integers(1_600_000_000_000, 1_600_100_000_000, n)),
        "IPV4_SRC_ADDR": rng.integers(0, 15, n),
        "IPV4_DST_ADDR": rng.integers(0, 15, n),
        "L4_SRC_PORT": rng.integers(1, 65535, n),
        "L4_DST_PORT": rng.choice([22, 80, 443, 53, 8080, 3306], n),
        "PROTOCOL": rng.choice([6, 17, 1], n),
        "L7_PROTO": rng.integers(0, 20, n),
        "IN_BYTES": rng.exponential(1000, n) + 40,
        "OUT_BYTES": rng.exponential(1000, n) + 40,
        "IN_PKTS": rng.integers(1, 100, n),
        "OUT_PKTS": rng.integers(1, 100, n),
        "FLOW_DURATION_MILLISECONDS": rng.exponential(5000, n) + 1,
        "TCP_FLAGS": rng.integers(0, 255, n),
        "CLIENT_TCP_FLAGS": rng.integers(0, 255, n),
        "SERVER_TCP_FLAGS": rng.integers(0, 255, n),
        "DURATION_IN": rng.exponential(2000, n),
        "DURATION_OUT": rng.exponential(2000, n),
        "MIN_TTL": rng.integers(30, 128, n),
        "MAX_TTL": rng.integers(30, 128, n),
        "LONGEST_FLOW_PKT": rng.integers(64, 1500, n),
        "SHORTEST_FLOW_PKT": rng.integers(40, 64, n),
        "MIN_IP_PKT_LEN": rng.integers(40, 64, n),
        "MAX_IP_PKT_LEN": rng.integers(64, 1500, n),
        "SRC_TO_DST_SECOND_BYTES": rng.exponential(500, n),
        "DST_TO_SRC_SECOND_BYTES": rng.exponential(500, n),
        "RETRANSMITTED_IN_BYTES": rng.exponential(10, n),
        "RETRANSMITTED_IN_PKTS": rng.integers(0, 5, n),
        "RETRANSMITTED_OUT_BYTES": rng.exponential(10, n),
        "RETRANSMITTED_OUT_PKTS": rng.integers(0, 5, n),
        "SRC_TO_DST_AVG_THROUGHPUT": rng.exponential(10000, n),
        "DST_TO_SRC_AVG_THROUGHPUT": rng.exponential(10000, n),
        "NUM_PKTS_UP_TO_128_BYTES": rng.integers(0, 50, n),
        "NUM_PKTS_128_TO_256_BYTES": rng.integers(0, 50, n),
        "NUM_PKTS_256_TO_512_BYTES": rng.integers(0, 50, n),
        "NUM_PKTS_512_TO_1024_BYTES": rng.integers(0, 50, n),
        "NUM_PKTS_1024_TO_1514_BYTES": rng.integers(0, 50, n),
        "TCP_WIN_MAX_IN": rng.integers(0, 65535, n),
        "TCP_WIN_MAX_OUT": rng.integers(0, 65535, n),
        "ICMP_TYPE": rng.integers(0, 20, n),
        "ICMP_IPV4_TYPE": rng.integers(0, 20, n),
        "DNS_QUERY_ID": rng.integers(0, 65535, n),
        "DNS_QUERY_TYPE": rng.integers(0, 20, n),
        "DNS_TTL_ANSWER": rng.integers(0, 128, n),
        "FTP_COMMAND_RET_CODE": rng.integers(0, 500, n),
        "SRC_TO_DST_IAT_MIN": rng.exponential(10, n),
        "SRC_TO_DST_IAT_MAX": rng.exponential(1000, n) + 100,
        "SRC_TO_DST_IAT_AVG": rng.exponential(100, n) + 10,
        "SRC_TO_DST_IAT_STDDEV": rng.exponential(50, n),
        "DST_TO_SRC_IAT_MIN": rng.exponential(10, n),
        "DST_TO_SRC_IAT_MAX": rng.exponential(1000, n) + 100,
        "DST_TO_SRC_IAT_AVG": rng.exponential(100, n) + 10,
        "DST_TO_SRC_IAT_STDDEV": rng.exponential(50, n),
        "Label": rng.integers(0, 2, n),
        "Attack": rng.choice(["Benign", "FTP-BruteForce", "Bot"], n),
    })
    return df.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)


def _build_raw_attack_fixture(seed: int = 0):
    """Fit a real FeaturePipeline, build a matching AnchorBinGraphSource + a
    small ArgusModel, and return everything needed to run A1/A4/A5 on one
    target flow, plus the target's own RAW row.
    """
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    label_to_id = {c: i for i, c in enumerate(class_names)}

    raw_df = _make_synthetic_raw_flows(n=300, seed=seed)
    pipeline = FeaturePipeline(protocol_topk=4, l7_proto_topk=4, dst_port_topk=4)
    pipeline.fit(raw_df)
    feature_names = pipeline.feature_names_
    transformed = pipeline.transform(raw_df)
    edge_features = transformed[feature_names].to_numpy(dtype=np.float32)

    times_ms = raw_df["FLOW_START_MILLISECONDS"].to_numpy()
    src_ids = raw_df["IPV4_SRC_ADDR"].to_numpy()
    dst_ids = raw_df["IPV4_DST_ADDR"].to_numpy()
    labels = raw_df["Attack"].map(label_to_id).to_numpy()

    source = AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, labels,
        anchor_bin_seconds=1, window_short_seconds=1, window_mid_seconds=30,
        window_long_seconds=300, neighbour_cap=8, seed=seed,
    )

    f_e = len(feature_names)
    cfg = load_config(overrides=["model.layers=1", "graph.neighbour_cap=8"])
    model = ArgusModel(cfg, f_e=f_e, f_v=18, class_names=class_names)

    bin_id, target_pos_in_short = None, None
    for b in source.unique_bins:
        batch = source.build_bin_batch(b, f_v=18)
        if batch is not None and batch["n_targets"] > 0:
            bin_id = b
            short_lo, short_hi = dict((bb, (lo, hi)) for bb, lo, hi in source.ranges["short"])[b]
            target_pos_in_short = np.nonzero(source.bin_ids[short_lo:short_hi] == b)[0][0] + short_lo
            break
    assert bin_id is not None

    batch = source.build_bin_batch(bin_id, f_v=18)
    raw_row = raw_df.iloc[[int(target_pos_in_short)]]

    return {
        "model": model, "pipeline": pipeline, "batch": batch, "raw_row": raw_row,
        "feature_names": feature_names, "device": torch.device("cpu"),
    }


def test_a1_epsilon_sweep_runs_and_respects_zero_epsilon():
    fx = _build_raw_attack_fixture(seed=1)
    results = run_a1_epsilon_sweep(
        fx["model"], fx["pipeline"], fx["batch"], fx["raw_row"], fx["feature_names"],
        target_index_within_bin=0, device=fx["device"], epsilons=[0.0, 0.1], steps=3, seed=0,
    )
    assert len(results) == 2
    assert results[0].epsilon == 0.0
    assert results[0].clean_decision == results[0].attacked_decision  # epsilon=0 => no-op
    for r in results:
        assert np.isfinite(r.clean_evidence_total)
        assert np.isfinite(r.attacked_evidence_total)


def test_a5_jitter_sweep_runs_and_reports_attacker_cost():
    fx = _build_raw_attack_fixture(seed=2)
    results = run_a5_jitter_sweep(
        fx["model"], fx["pipeline"], fx["batch"], fx["raw_row"], fx["feature_names"],
        target_index_within_bin=0, device=fx["device"], sigmas=[0.0, 0.5], seed=0,
    )
    assert len(results) == 2
    assert results[0].sigma == 0.0
    assert results[0].duration_change_frac == 0.0
    for r in results:
        assert np.isfinite(r.clean_evidence_total)
        assert np.isfinite(r.attacked_evidence_total)


def test_a4_adaptive_runs_both_objectives():
    fx = _build_raw_attack_fixture(seed=3)
    for objective in ("evasion", "unknown_avoidance"):
        result = run_a4_adaptive(
            fx["model"], fx["pipeline"], fx["batch"], fx["raw_row"], fx["feature_names"],
            target_index_within_bin=0, device=fx["device"], objective=objective, steps=3, seed=0,
        )
        assert result.objective == objective
        assert np.isfinite(result.attacked_cos_benign)
        assert np.isfinite(result.attacked_evidence_total)


def test_a3_ema_update_moves_prototype_only_when_gate_open():
    torch.manual_seed(0)
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    head = EPCHead(d_h=32, d_z=16, class_names=class_names)
    class_idx = class_names.index("Bot")

    gated = run_a3_poison_sweep(
        head, class_idx, poison_rate=1.0, momentum=0.9, n_steps=20,
        theta_unknown=1e9, gate_enabled=True, seed=0,  # impossible threshold => gate always blocks
    )
    assert gated.accepted_fraction == 0.0
    assert gated.final_drift == 0.0

    ungated = run_a3_poison_sweep(
        head, class_idx, poison_rate=1.0, momentum=0.5, n_steps=20,
        theta_unknown=1e9, gate_enabled=False, seed=0,
    )
    assert ungated.final_drift > 0.0


def test_craft_poison_embedding_is_unit_norm():
    torch.manual_seed(0)
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    head = EPCHead(d_h=32, d_z=16, class_names=class_names)
    z = craft_poison_embedding(head, class_idx=1, seed=0)
    assert torch.isfinite(z).all()
    assert abs(float(torch.norm(z).item()) - 1.0) < 1e-4
