"""Validate constants and feature partition completeness."""

import pytest

from argus.constants import RAW_FEATURE_COLS
from argus.features.partition import assert_partition_complete


def test_raw_feature_count():
    assert len(RAW_FEATURE_COLS) == 53


def test_partition_complete_for_emitted_features():
    # The 147-dim edge vector from the spec. We list the canonical names that
    # would be emitted by the feature pipeline.
    features = []
    # TE1 heavy-tail (23)
    te1 = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
    ]
    features.extend(te1)
    # Bounded numeric (14)
    bounded = [
        "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
        "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
        "FTP_COMMAND_RET_CODE",
    ]
    features.extend(bounded)
    # Packet-size histogram + simplex (6)
    features.extend([
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES", "pkt_size_hist_sum_log",
    ])
    # TCP flags (24)
    for prefix in ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]:
        for bit in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]:
            features.append(f"{prefix}_{bit}")
    # Protocols (placeholder prefix)
    features.extend(["PROTOCOL_0", "L7_PROTO_0"])
    # Ports (41)
    features.extend([
        "L4_SRC_PORT_is_well_known", "L4_SRC_PORT_is_registered", "L4_SRC_PORT_is_ephemeral", "L4_SRC_PORT_log",
        "L4_DST_PORT_is_well_known", "L4_DST_PORT_is_registered", "L4_DST_PORT_is_ephemeral", "L4_DST_PORT_log",
        "L4_DST_PORT_topk_22", "L4_DST_PORT_topk_80",
    ])
    # TE2 (13)
    features.extend([
        "iat_cv_fwd", "iat_cv_bwd", "iat_burst_fwd", "iat_burst_bwd",
        "duty_cycle", "pkt_rate", "byte_rate", "dir_asymmetry",
        "pkt_size_spread", "bytes_per_pkt_in", "bytes_per_pkt_out",
        "retrans_ratio", "iat_undefined",
    ])
    a_idx, b_idx = assert_partition_complete(features)
    assert len(a_idx) + len(b_idx) == len(features)


def test_partition_rejects_unknown():
    with pytest.raises(ValueError):
        assert_partition_complete(["MYSTERY_FEATURE"])
