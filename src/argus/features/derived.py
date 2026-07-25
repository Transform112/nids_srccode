"""TE2 — derived rhythm descriptors.

See docs/03_FEATURE_ENGINEERING.md §4. Twelve features + iat_undefined guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def compute_te2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 13 TE2 derived features from raw NF-v3 columns."""
    out = pd.DataFrame(index=df.index)

    out["iat_cv_fwd"] = df["SRC_TO_DST_IAT_STDDEV"] / (df["SRC_TO_DST_IAT_AVG"] + EPS)
    out["iat_cv_bwd"] = df["DST_TO_SRC_IAT_STDDEV"] / (df["DST_TO_SRC_IAT_AVG"] + EPS)
    out["iat_burst_fwd"] = (df["SRC_TO_DST_IAT_MAX"] - df["SRC_TO_DST_IAT_MIN"]) / (
        df["SRC_TO_DST_IAT_AVG"] + EPS
    )
    out["iat_burst_bwd"] = (df["DST_TO_SRC_IAT_MAX"] - df["DST_TO_SRC_IAT_MIN"]) / (
        df["DST_TO_SRC_IAT_AVG"] + EPS
    )
    out["duty_cycle"] = (df["DURATION_IN"] + df["DURATION_OUT"]) / (
        df["FLOW_DURATION_MILLISECONDS"] + EPS
    )
    out["pkt_rate"] = (df["IN_PKTS"] + df["OUT_PKTS"]) / (df["FLOW_DURATION_MILLISECONDS"] + EPS)
    out["byte_rate"] = (df["IN_BYTES"] + df["OUT_BYTES"]) / (df["FLOW_DURATION_MILLISECONDS"] + EPS)
    out["dir_asymmetry"] = (df["SRC_TO_DST_IAT_AVG"] - df["DST_TO_SRC_IAT_AVG"]) / (
        df["SRC_TO_DST_IAT_AVG"] + df["DST_TO_SRC_IAT_AVG"] + EPS
    )
    out["pkt_size_spread"] = df["LONGEST_FLOW_PKT"] - df["SHORTEST_FLOW_PKT"]
    out["bytes_per_pkt_in"] = df["IN_BYTES"] / (df["IN_PKTS"] + EPS)
    out["bytes_per_pkt_out"] = df["OUT_BYTES"] / (df["OUT_PKTS"] + EPS)
    out["retrans_ratio"] = (df["RETRANSMITTED_IN_PKTS"] + df["RETRANSMITTED_OUT_PKTS"]) / (
        df["IN_PKTS"] + df["OUT_PKTS"] + EPS
    )
    out["iat_undefined"] = (df["IN_PKTS"] + df["OUT_PKTS"] <= 2).astype(float)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
