"""NF-v3 column lists and dataset metadata.

All column names match the raw CSV header exactly. Never index positionally.
"""

RAW_FEATURE_COLS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

LABEL_COLS = ["Label", "Attack"]

# 10 temporal features added in NF-v3 (includes the two timestamps).
TEMPORAL_FEATURES = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

# Columns used for node identity construction.
NODE_IDENTITY_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]

# Heavy-tailed numeric columns that receive TE1 signed-log + quantile conditioning.
TE1_COLUMNS = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

# Bounded numeric columns: robust scale only.
BOUNDED_NUMERIC = [
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

# Packet-size histogram bins.
PKT_SIZE_BINS = [
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]

# TCP flag bitfields expanded to 24 binary columns.
TCP_FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
TCP_FLAG_BITS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]

# Dataset metadata measured from the real CSVs (2026-07-25).
DATASET_STATS = {
    "cicids2018": {
        "rows": 20_115_529,
        "src_ips": 181_876,
        "attack_classes": 14,
        "benign_fraction": 0.8707,
        "node_granularity": "ip",
    },
    "ton_iot": {
        "rows": 27_520_260,
        "src_ips": 15_270,
        "attack_classes": 9,
        "benign_fraction": 0.6102,
        "node_granularity": "ip",
    },
    "unsw_nb15": {
        "rows": 2_365_424,
        "src_ips": 40,
        "attack_classes": 9,
        "benign_fraction": 0.9460,
        "node_granularity": "ip_port",
    },
    "bot_iot": {
        "rows": 16_933_808,
        "src_ips": 20,
        "attack_classes": 4,
        "benign_fraction": 0.0031,
        "node_granularity": "ip_port",
    },
}

MIN_UNIQUE_SRC_IP = 1_000

# Label canonicalisation (docs/02_DATASETS.md §5.2). Raises on unrecognised labels.
CANONICAL_LABELS: dict[str, dict[str, str]] = {
    "cicids2018": {
        "Benign": "benign",
        "DDOS_attack-HOIC": "ddos_hoic",
        "DDOS_attack-LOIC-UDP": "ddos_loic_udp",
        "DDoS_attacks-LOIC-HTTP": "ddos_loic_http",
        "DoS_attacks-Hulk": "dos_hulk",
        "DoS_attacks-GoldenEye": "dos_goldeneye",
        "DoS_attacks-Slowloris": "dos_slowloris",
        "DoS_attacks-SlowHTTPTest": "dos_slowhttptest",
        "FTP-BruteForce": "brute_ftp",
        "SSH-Bruteforce": "brute_ssh",
        "Brute_Force_-Web": "brute_web",
        "Brute_Force_-XSS": "brute_xss",
        "SQL_Injection": "sql_injection",
        "Infilteration": "infiltration",  # sic — misspelled in source data
        "Bot": "bot",
    },
    "ton_iot": {
        "Benign": "benign",
        "Backdoor": "backdoor",
        "ddos": "ddos",
        "dos": "dos",
        "injection": "injection",
        "mitm": "mitm",
        "password": "password",
        "ransomware": "ransomware",
        "scanning": "scanning",
        "xss": "xss",
    },
    "unsw_nb15": {
        "Benign": "benign",
        "Exploits": "exploits",
        "Fuzzers": "fuzzers",
        "Generic": "generic",
        "Reconnaissance": "reconnaissance",
        "DoS": "dos",
        "Backdoor": "backdoor",
        "Shellcode": "shellcode",
        "Analysis": "analysis",
        "Worms": "worms",
    },
    "bot_iot": {
        "Benign": "benign",
        "DoS": "dos",
        "DDoS": "ddos",
        "Reconnaissance": "reconnaissance",
        "Theft": "theft",
    },
}

# Family membership for CICIDS2018 (docs/02_DATASETS.md §2.3). Used for Protocol B2.
FAMILY_OF: dict[str, str] = {
    "dos_hulk": "dos",
    "dos_goldeneye": "dos",
    "dos_slowloris": "dos",
    "dos_slowhttptest": "dos",
    "ddos_hoic": "ddos",
    "ddos_loic_udp": "ddos",
    "ddos_loic_http": "ddos",
    "brute_ftp": "brute",
    "brute_ssh": "brute",
    "brute_web": "brute",
    "brute_xss": "brute",
    "sql_injection": "web",
    "infiltration": "other",
    "bot": "other",
}


def canonicalise(dataset: str, raw_label: str) -> str:
    """Map a raw label to its canonical form. Raises on unrecognised labels."""
    mapping = CANONICAL_LABELS.get(dataset)
    if mapping is None:
        raise ValueError(f"Unknown dataset '{dataset}' for label canonicalisation")
    if raw_label not in mapping:
        raise ValueError(
            f"Unrecognised label '{raw_label}' for dataset '{dataset}'. "
            "The label vocabulary is stale; update CANONICAL_LABELS."
        )
    return mapping[raw_label]


# Measured active-day -> attack mapping (docs/02_DATASETS.md §4). Canonical labels.
DAY_ATTACK_MAP: dict[str, dict[int, list[str]]] = {
    "cicids2018": {
        1: ["brute_ftp", "brute_ssh"],
        2: ["dos_goldeneye", "dos_slowloris"],
        3: ["dos_hulk", "dos_slowhttptest"],
        4: ["ddos_loic_http"],
        5: ["ddos_hoic", "ddos_loic_udp"],
        6: ["brute_web", "brute_xss", "sql_injection"],
        7: [],
        8: [],
        9: ["infiltration"],
        10: ["infiltration"],
        11: ["bot"],
    },
    "ton_iot": {
        1: ["scanning"],
        2: ["dos", "scanning"],
        3: ["ddos", "dos", "injection"],
        4: ["ddos", "password"],
        5: ["password", "xss"],
        6: ["backdoor", "ransomware"],
        7: ["backdoor", "mitm"],
    },
}

# Minority-tier thresholds (docs/02_DATASETS.md §6.1, §6.3).
MINORITY_THRESHOLD = 200_000
MIN_COUNT_FOR_PROTOTYPE = 100
