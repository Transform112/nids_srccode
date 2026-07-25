"""End-to-end feature pipeline test on synthetic NF-v3-shaped data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.features.pipeline import FeaturePipeline


def _make_synthetic_raw(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "FLOW_START_MILLISECONDS": np.sort(rng.integers(1_600_000_000_000, 1_600_100_000_000, n)),
            "FLOW_END_MILLISECONDS": np.sort(rng.integers(1_600_000_000_000, 1_600_100_000_000, n)),
            "IPV4_SRC_ADDR": rng.integers(0, 20, n),
            "IPV4_DST_ADDR": rng.integers(0, 20, n),
            "L4_SRC_PORT": rng.integers(1, 65535, n),
            "L4_DST_PORT": rng.choice([22, 80, 443, 53, 8080, 3306], n),
            "PROTOCOL": rng.choice([6, 17, 1], n),
            "L7_PROTO": rng.integers(0, 20, n),
            "IN_BYTES": rng.exponential(1000, n),
            "OUT_BYTES": rng.exponential(1000, n),
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
            "SRC_TO_DST_IAT_MAX": rng.exponential(1000, n),
            "SRC_TO_DST_IAT_AVG": rng.exponential(100, n),
            "SRC_TO_DST_IAT_STDDEV": rng.exponential(50, n),
            "DST_TO_SRC_IAT_MIN": rng.exponential(10, n),
            "DST_TO_SRC_IAT_MAX": rng.exponential(1000, n),
            "DST_TO_SRC_IAT_AVG": rng.exponential(100, n),
            "DST_TO_SRC_IAT_STDDEV": rng.exponential(50, n),
            "Label": rng.integers(0, 2, n),
            "Attack": rng.choice(["Benign", "FTP-BruteForce", "Bot"], n),
        }
    )
    return df


def test_pipeline_fit_transform_partition_complete():
    df = _make_synthetic_raw(n=500)
    pipeline = FeaturePipeline(protocol_topk=8, l7_proto_topk=16, dst_port_topk=32)
    pipeline.fit(df)
    out = pipeline.transform(df)
    assert len(out) == len(df)
    assert not out.isna().any().any()
    a_idx, b_idx = pipeline.assert_channels()
    assert len(a_idx) + len(b_idx) == len(pipeline.feature_names_)


def test_pipeline_fit_on_train_only_no_leakage():
    df = _make_synthetic_raw(n=500)
    train, test = df.iloc[:400], df.iloc[400:]
    pipeline = FeaturePipeline()
    pipeline.fit(train)
    out_test = pipeline.transform(test)
    assert len(out_test) == len(test)


def test_pipeline_include_temporal_block_false_drops_all_temporal_columns():
    """Temporal ladder L0 rung: no temporal columns at all, not just unconditioned."""
    from argus.constants import TE1_COLUMNS

    df = _make_synthetic_raw(n=300)
    with_temporal = FeaturePipeline()
    with_temporal.fit(df)

    without_temporal = FeaturePipeline(include_temporal_block=False)
    without_temporal.fit(df)
    out = without_temporal.transform(df)

    assert len(without_temporal.feature_names_) < len(with_temporal.feature_names_)
    for col in TE1_COLUMNS:
        assert col not in out.columns
    assert "iat_undefined" not in out.columns
    assert "iat_cv_fwd" not in out.columns
    assert not out.isna().any().any()


def test_pipeline_save_load(tmp_path):
    df = _make_synthetic_raw(n=200)
    pipeline = FeaturePipeline()
    pipeline.fit(df)
    save_path = tmp_path / "pipeline.joblib"
    pipeline.save(save_path)
    loaded = FeaturePipeline.load(save_path)
    out1 = pipeline.transform(df)
    out2 = loaded.transform(df)
    pd.testing.assert_frame_equal(out1, out2)
    assert (tmp_path / "feature_manifest.json").exists()
