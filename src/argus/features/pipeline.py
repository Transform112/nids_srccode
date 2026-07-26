"""Full feature pipeline: fit on train only, transform any split, emit manifest.

See docs/03_FEATURE_ENGINEERING.md §7 for the final vector layout (F_e = 147).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from argus.constants import BOUNDED_NUMERIC, TE1_COLUMNS
from argus.features.conditioning import TE1Conditioner
from argus.features.derived import compute_te2
from argus.features.encoders import (
    TopKOneHotEncoder,
    expand_tcp_flags,
    packet_size_histogram,
    port_features,
)
from argus.features.partition import assert_partition_complete

# TE2 numeric columns that get signed-log+quantile conditioning (all except the mask).
TE2_CONDITIONED_COLS = [
    "iat_cv_fwd", "iat_cv_bwd", "iat_burst_fwd", "iat_burst_bwd",
    "duty_cycle", "pkt_rate", "byte_rate", "dir_asymmetry",
    "pkt_size_spread", "bytes_per_pkt_in", "bytes_per_pkt_out", "retrans_ratio",
]


class FeaturePipeline:
    """Fit/transform/save/load the full F_e=147 edge feature vector."""

    def __init__(
        self,
        protocol_topk: int = 8,
        l7_proto_topk: int = 16,
        dst_port_topk: int = 32,
        quantile_n: int = 1000,
        quantile_subsample: int = 1_000_000,
        clip_post_transform: float = 5.0,
        te1_enabled: bool = True,
        te2_enabled: bool = True,
        include_temporal_block: bool = True,
    ) -> None:
        self.te1_enabled = te1_enabled
        self.te2_enabled = te2_enabled
        self.include_temporal_block = include_temporal_block
        # Same clip the TE1/TE2 conditioners apply, but for the bounded block.
        self.clip_post_transform = clip_post_transform
        self.te1 = TE1Conditioner(
            n_quantiles=quantile_n, subsample=quantile_subsample, clip=clip_post_transform
        )
        self.te2_cond = TE1Conditioner(
            n_quantiles=quantile_n,
            subsample=quantile_subsample,
            clip=clip_post_transform,
            columns=list(TE2_CONDITIONED_COLS),
        )
        self.bounded_scaler = RobustScaler()
        self.protocol_enc = TopKOneHotEncoder("PROTOCOL", protocol_topk, prefix="PROTOCOL")
        self.l7_enc = TopKOneHotEncoder("L7_PROTO", l7_proto_topk, prefix="L7_PROTO")
        self.dst_port_enc = TopKOneHotEncoder(
            "L4_DST_PORT", dst_port_topk, prefix="L4_DST_PORT_topk"
        )
        self.feature_names_: list[str] = []
        self.split_hash_: str | None = None

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        if self.te1_enabled and self.include_temporal_block:
            self.te1.fit(df)
        te2 = compute_te2(df)
        if self.te2_enabled and self.include_temporal_block:
            self.te2_cond.fit(te2)
        self.bounded_scaler.fit(df[BOUNDED_NUMERIC].fillna(0).to_numpy(dtype=float))
        self.protocol_enc.fit(df)
        self.l7_enc.fit(df)
        self.dst_port_enc.fit(df)
        # Run transform once to lock in feature_names_.
        self.transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []

        if self.include_temporal_block:
            if self.te1_enabled:
                blocks.append(self.te1.transform(df))
            else:
                blocks.append(df[TE1_COLUMNS].astype(float))

        bounded_arr = self.bounded_scaler.transform(
            df[BOUNDED_NUMERIC].fillna(0).to_numpy(dtype=float)
        )
        # Clip to the same range as the TE1/TE2 conditioners. RobustScaler leaves
        # columns whose IQR is 0 (the >75%-zeros sparse columns: DNS_QUERY_ID,
        # ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE, DNS_TTL_ANSWER) at scale_=1,
        # i.e. a raw pass-through — DNS_QUERY_ID then reaches the model at up to
        # 65535. Without this clip the bounded block silently escaped the ±clip
        # bound every other numeric block obeys (see docs/BUGS.md).
        if self.clip_post_transform is not None:
            np.clip(
                bounded_arr,
                -self.clip_post_transform,
                self.clip_post_transform,
                out=bounded_arr,
            )
        bounded = pd.DataFrame(bounded_arr, columns=BOUNDED_NUMERIC, index=df.index)
        blocks.append(bounded)

        blocks.append(packet_size_histogram(df))
        blocks.append(expand_tcp_flags(df))
        blocks.append(self.protocol_enc.transform(df))
        blocks.append(self.l7_enc.transform(df))
        blocks.append(port_features(df, "L4_SRC_PORT"))
        blocks.append(port_features(df, "L4_DST_PORT"))
        blocks.append(self.dst_port_enc.transform(df))

        te2 = compute_te2(df)
        if self.include_temporal_block:
            if self.te2_enabled:
                conditioned = self.te2_cond.transform(te2)
                te2_out = pd.concat([conditioned, te2[["iat_undefined"]]], axis=1)
            else:
                te2_out = te2
            blocks.append(te2_out)

        result = pd.concat(blocks, axis=1)
        self.feature_names_ = list(result.columns)
        return result

    def assert_channels(self) -> tuple[list[int], list[int]]:
        """Validate the provenance partition is complete for the fitted feature set."""
        return assert_partition_complete(self.feature_names_)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        manifest = {
            "feature_names": self.feature_names_,
            "f_e": len(self.feature_names_),
            "split_hash": self.split_hash_,
        }
        with open(path.parent / "feature_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        return joblib.load(path)
