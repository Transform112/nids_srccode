"""Provenance partition of edge features into Channel A (controllable) and Channel B (observer).

The partition is a static table; every emitted edge feature must belong to exactly one channel.
See docs/03_FEATURE_ENGINEERING.md §6.
"""

from __future__ import annotations

# Channel A: attacker-controllable / forgeable without degrading the attack.
CONTROLLABLE: frozenset[str] = frozenset({
    # Volume / packet counts
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    # TCP flags
    "TCP_FLAGS_FIN", "TCP_FLAGS_SYN", "TCP_FLAGS_RST", "TCP_FLAGS_PSH",
    "TCP_FLAGS_ACK", "TCP_FLAGS_URG", "TCP_FLAGS_ECE", "TCP_FLAGS_CWR",
    "CLIENT_TCP_FLAGS_FIN", "CLIENT_TCP_FLAGS_SYN", "CLIENT_TCP_FLAGS_RST", "CLIENT_TCP_FLAGS_PSH",
    "CLIENT_TCP_FLAGS_ACK", "CLIENT_TCP_FLAGS_URG", "CLIENT_TCP_FLAGS_ECE", "CLIENT_TCP_FLAGS_CWR",
    "SERVER_TCP_FLAGS_FIN", "SERVER_TCP_FLAGS_SYN", "SERVER_TCP_FLAGS_RST", "SERVER_TCP_FLAGS_PSH",
    "SERVER_TCP_FLAGS_ACK", "SERVER_TCP_FLAGS_URG", "SERVER_TCP_FLAGS_ECE", "SERVER_TCP_FLAGS_CWR",
    # Ports
    "L4_SRC_PORT_is_well_known", "L4_SRC_PORT_is_registered", "L4_SRC_PORT_is_ephemeral", "L4_SRC_PORT_log",
    "L4_DST_PORT_is_well_known", "L4_DST_PORT_is_registered", "L4_DST_PORT_is_ephemeral", "L4_DST_PORT_log",
    # dst port top-k one-hot columns are generated dynamically; placeholder prefix
    "L4_DST_PORT_topk_",
    # Protocols
    "PROTOCOL_", "L7_PROTO_",
    # Packet-size histogram + derived simplex
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "pkt_size_hist_norm_", "pkt_size_hist_sum_log",
    # Length-like / bounded-but-easily-set
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    # DNS / ICMP / FTP
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE",
    # Derived from sizes
    "pkt_size_spread", "bytes_per_pkt_in", "bytes_per_pkt_out",
})

# Channel B: observer-derived / costly to forge or emergent from network path.
OBSERVER: frozenset[str] = frozenset({
    # IAT statistics
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
    # Durations and throughput
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    # Path / network emergent
    "MIN_TTL", "MAX_TTL",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "DNS_TTL_ANSWER",
    # TE2 derived rhythm descriptors
    "iat_cv_fwd", "iat_cv_bwd", "iat_burst_fwd", "iat_burst_bwd",
    "duty_cycle", "pkt_rate", "byte_rate", "dir_asymmetry", "retrans_ratio", "iat_undefined",
})


def channel_of(name: str) -> str | None:
    """Return 'A', 'B', or None for a feature name. Handles dynamic prefixes."""
    if name in CONTROLLABLE:
        return "A"
    if name in OBSERVER:
        return "B"
    # Prefix matches for dynamic columns
    for prefix in ("L4_DST_PORT_topk_", "PROTOCOL_", "L7_PROTO_", "pkt_size_hist_norm_"):
        if name.startswith(prefix):
            return "A"
    return None


def assert_partition_complete(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Return (channel_a_indices, channel_b_indices) after asserting completeness.

    Raises:
        ValueError: if any feature is missing from both channels or appears in both.
    """
    a_idx, b_idx = [], []
    for i, name in enumerate(feature_names):
        ch = channel_of(name)
        if ch is None:
            raise ValueError(f"Feature '{name}' is not assigned to any provenance channel")
        if ch == "A":
            a_idx.append(i)
        else:
            b_idx.append(i)
    # No overlap because channel_of returns a single value and sets are disjoint.
    return a_idx, b_idx
