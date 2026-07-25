"""Node identity assignment and graph builder with the TRAP-1 guard.

See docs/04_GRAPH_CONSTRUCTION.md §1 and docs/02_DATASETS.md §3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argus.constants import MIN_UNIQUE_SRC_IP


def assign_node_ids(
    df: pd.DataFrame,
    node_granularity: str = "ip",
    src_ip_col: str = "IPV4_SRC_ADDR",
    dst_ip_col: str = "IPV4_DST_ADDR",
    src_port_col: str = "L4_SRC_PORT",
    dst_port_col: str = "L4_DST_PORT",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map each flow's (src, dst) to global integer node ids.

    Returns:
        src_ids: [N] int64
        dst_ids: [N] int64
        id_map: dict node_key -> int id
    """
    if node_granularity == "ip":
        src_key = df[src_ip_col].astype(str)
        dst_key = df[dst_ip_col].astype(str)
    elif node_granularity == "ip_port":
        src_key = df[src_ip_col].astype(str) + ":" + df[src_port_col].astype(str)
        dst_key = df[dst_ip_col].astype(str) + ":" + df[dst_port_col].astype(str)
    else:
        raise ValueError(f"Unknown node_granularity: {node_granularity}")

    all_keys = pd.concat([src_key, dst_key], ignore_index=True)
    categories = pd.Categorical(all_keys)
    id_map = {cat: i for i, cat in enumerate(categories.categories)}
    n = len(df)
    src_ids = categories.codes[:n]
    dst_ids = categories.codes[n:]
    return src_ids.astype(np.int64), dst_ids.astype(np.int64), id_map


def enforce_trap1_guard(
    node_granularity: str,
    measured_unique_src_ip: int,
    min_unique_src_ip: int = MIN_UNIQUE_SRC_IP,
) -> None:
    """Hard-fail if IP-level nodes are requested on a dataset with too few source IPs."""
    if node_granularity == "ip" and measured_unique_src_ip < min_unique_src_ip:
        raise ValueError(
            f"TRAP-1: {measured_unique_src_ip} unique source IPs < {min_unique_src_ip}. "
            "Use node_granularity='ip_port' for this dataset (docs/02_DATASETS.md §3)."
        )
