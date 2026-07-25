"""Cleaning: nulls, negative artefacts, deduplication.

See docs/02_DATASETS.md §5.1.
"""

from __future__ import annotations

import pandas as pd

from argus.constants import RAW_FEATURE_COLS

# nProbe artefact columns that can contain spurious negative values.
NEGATIVE_CLIP_COLS = [
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

MIN_VALID_MS = 1_388_534_400_000  # 2014-01-01
MAX_VALID_MS = 1_577_836_800_000  # 2020-01-01


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cleaning rules; return the cleaned frame and a report of counts removed.

    Steps (docs/02_DATASETS.md §5.1):
        1. Drop rows with null Attack or null FLOW_START_MILLISECONDS.
        2. Clip negative durations/IAT to 0.
        3. Deduplicate on the full feature tuple.
        4. Cast timestamps to int64 and verify plausible range.
    """
    report: dict[str, int] = {"input_rows": len(df)}

    before = len(df)
    df = df.dropna(subset=["Attack", "FLOW_START_MILLISECONDS"])
    report["dropped_null"] = before - len(df)

    for col in NEGATIVE_CLIP_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    before = len(df)
    dedup_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)
    report["dropped_duplicates"] = before - len(df)

    df["FLOW_START_MILLISECONDS"] = df["FLOW_START_MILLISECONDS"].astype("int64")
    df["FLOW_END_MILLISECONDS"] = df["FLOW_END_MILLISECONDS"].astype("int64")

    before = len(df)
    plausible = df["FLOW_START_MILLISECONDS"].between(MIN_VALID_MS, MAX_VALID_MS)
    df = df[plausible]
    report["dropped_implausible_time"] = before - len(df)

    report["output_rows"] = len(df)
    return df.reset_index(drop=True), report
