"""Domain feasibility projection for raw NF-v3 feature vectors.

See docs/10_ADVERSARIAL.md §1.3. Every perturbed feature vector must satisfy
these constraints; applied after every attack optimisation step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HEADER_BYTES = 40  # minimum IPv4+TCP header size


def project(df: pd.DataFrame) -> pd.DataFrame:
    """Project a raw-space (untransformed) feature dataframe onto the feasible set.

    Operates column-wise on whichever of the constrained columns are present;
    silently skips columns that are absent (e.g. a reduced feature subset).
    """
    df = df.copy()

    if "IN_PKTS" in df.columns:
        df["IN_PKTS"] = df["IN_PKTS"].clip(lower=0).round()
    if "OUT_PKTS" in df.columns:
        df["OUT_PKTS"] = df["OUT_PKTS"].clip(lower=0).round()

    if "IN_BYTES" in df.columns and "IN_PKTS" in df.columns:
        df["IN_BYTES"] = np.maximum(df["IN_BYTES"], df["IN_PKTS"] * MIN_HEADER_BYTES)
    if "OUT_BYTES" in df.columns and "OUT_PKTS" in df.columns:
        df["OUT_BYTES"] = np.maximum(df["OUT_BYTES"], df["OUT_PKTS"] * MIN_HEADER_BYTES)

    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        df["FLOW_DURATION_MILLISECONDS"] = df["FLOW_DURATION_MILLISECONDS"].clip(lower=0)
        if "DURATION_IN" in df.columns and "DURATION_OUT" in df.columns:
            total = df["DURATION_IN"] + df["DURATION_OUT"]
            over = total > df["FLOW_DURATION_MILLISECONDS"]
            if over.any():
                scale = (df["FLOW_DURATION_MILLISECONDS"] / total.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
                df.loc[over, "DURATION_IN"] = df.loc[over, "DURATION_IN"] * scale[over]
                df.loc[over, "DURATION_OUT"] = df.loc[over, "DURATION_OUT"] * scale[over]

    if "SHORTEST_FLOW_PKT" in df.columns and "LONGEST_FLOW_PKT" in df.columns:
        lo = np.minimum(df["SHORTEST_FLOW_PKT"], df["LONGEST_FLOW_PKT"])
        hi = np.maximum(df["SHORTEST_FLOW_PKT"], df["LONGEST_FLOW_PKT"]).clip(upper=1514)
        df["SHORTEST_FLOW_PKT"], df["LONGEST_FLOW_PKT"] = lo, hi

    if "MIN_IP_PKT_LEN" in df.columns and "MAX_IP_PKT_LEN" in df.columns:
        lo = np.minimum(df["MIN_IP_PKT_LEN"], df["MAX_IP_PKT_LEN"])
        hi = np.maximum(df["MIN_IP_PKT_LEN"], df["MAX_IP_PKT_LEN"])
        df["MIN_IP_PKT_LEN"], df["MAX_IP_PKT_LEN"] = lo, hi

    for prefix in ("SRC_TO_DST_IAT", "DST_TO_SRC_IAT"):
        min_c, avg_c, max_c, std_c = f"{prefix}_MIN", f"{prefix}_AVG", f"{prefix}_MAX", f"{prefix}_STDDEV"
        if all(c in df.columns for c in (min_c, avg_c, max_c)):
            lo = df[[min_c, avg_c, max_c]].min(axis=1)
            hi = df[[min_c, avg_c, max_c]].max(axis=1)
            mid = df[avg_c].clip(lower=lo, upper=hi)
            df[min_c], df[avg_c], df[max_c] = lo, mid, hi
        if std_c in df.columns:
            df[std_c] = df[std_c].clip(lower=0)

    for col in df.columns:
        if col.endswith("_TCP_FLAGS") or col == "TCP_FLAGS":
            continue  # bitfields handled upstream of one-hot expansion

    if "L4_SRC_PORT" in df.columns:
        df["L4_SRC_PORT"] = df["L4_SRC_PORT"].clip(lower=0, upper=65535).round()
    if "L4_DST_PORT" in df.columns:
        df["L4_DST_PORT"] = df["L4_DST_PORT"].clip(lower=0, upper=65535).round()

    return df


def assert_feasible(df: pd.DataFrame) -> None:
    """Raise if any row violates a checkable constraint. Used in tests."""
    if "IN_BYTES" in df.columns and "IN_PKTS" in df.columns:
        assert (df["IN_BYTES"] >= df["IN_PKTS"] * MIN_HEADER_BYTES - 1e-6).all(), "IN_BYTES below header floor"
    if "OUT_BYTES" in df.columns and "OUT_PKTS" in df.columns:
        assert (df["OUT_BYTES"] >= df["OUT_PKTS"] * MIN_HEADER_BYTES - 1e-6).all(), "OUT_BYTES below header floor"
    for prefix in ("SRC_TO_DST_IAT", "DST_TO_SRC_IAT"):
        min_c, avg_c, max_c = f"{prefix}_MIN", f"{prefix}_AVG", f"{prefix}_MAX"
        if all(c in df.columns for c in (min_c, avg_c, max_c)):
            assert (df[min_c] <= df[avg_c] + 1e-6).all(), f"{min_c} > {avg_c}"
            assert (df[avg_c] <= df[max_c] + 1e-6).all(), f"{avg_c} > {max_c}"
