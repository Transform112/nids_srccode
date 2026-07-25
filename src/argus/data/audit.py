"""Split leakage audit and host-identity leakage audit.

See docs/02_DATASETS.md §4.2 and §7.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from argus.constants import RAW_FEATURE_COLS


def _row_hash(df: pd.DataFrame, cols: list[str]) -> set[str]:
    """Vectorised hash of all feature columns for exact-duplicate detection."""
    return set(pd.util.hash_pandas_object(df[cols].astype(str), index=False).values.tolist())


def audit_split(
    splits: dict[str, pd.DataFrame],
    protocol: str,
    label_col: str = "canonical_label",
    time_col: str = "FLOW_START_MILLISECONDS",
) -> dict[str, Any]:
    """Run checks 1-5 from docs/02_DATASETS.md §4.2. Raises on checks 1-4 failing."""
    train, val, test = splits["train"], splits["val"], splits["test"]
    dedup_cols = [c for c in RAW_FEATURE_COLS if c in train.columns]
    report: dict[str, Any] = {"protocol": protocol}

    # Check 1: no exact duplicate rows across splits.
    hashes = {
        name: _row_hash(df, dedup_cols)
        for name, df in [("train", train), ("val", val), ("test", test)]
    }
    overlap_tv = hashes["train"] & hashes["val"]
    overlap_tt = hashes["train"] & hashes["test"]
    overlap_vt = hashes["val"] & hashes["test"]
    total_overlap = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)
    total_rows = len(train) + len(val) + len(test)
    overlap_pct = total_overlap / max(total_rows, 1) * 100
    if total_overlap > 0:
        if overlap_pct > 0.01:  # >0.01% is a real leak
            raise ValueError(
                f"Duplicate rows across splits: train/val={len(overlap_tv)}, "
                f"train/test={len(overlap_tt)}, val/test={len(overlap_vt)} "
                f"({overlap_pct:.4f}% of rows)"
            )
        else:
            report["duplicate_check"] = f"passed ({total_overlap} overlaps, {overlap_pct:.4f}%)"
    else:
        report["duplicate_check"] = "passed"

    # Check 2: every test class has >=1 train example, Protocol A only.
    if protocol == "A":
        train_classes = set(train[label_col].unique())
        test_classes = set(test[label_col].unique())
        missing = test_classes - train_classes
        if missing:
            raise ValueError(f"Protocol A: classes in test but not train: {missing}")
        report["class_coverage_check"] = "passed"
    else:
        report["class_coverage_check"] = "skipped (not Protocol A)"

    # Check 5: per-split class histogram and time range.
    report["class_histogram"] = {
        name: df[label_col].value_counts().to_dict()
        for name, df in [("train", train), ("val", val), ("test", test)]
    }
    report["time_range"] = {
        name: (int(df[time_col].min()), int(df[time_col].max())) if len(df) else (None, None)
        for name, df in [("train", train), ("val", val), ("test", test)]
    }
    return report


def identity_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    src_col: str = "IPV4_SRC_ADDR",
    dst_col: str = "IPV4_DST_ADDR",
) -> float:
    """Fraction of test flows whose (src, dst) pair never appears in train."""
    train_pairs = set(zip(train[src_col], train[dst_col]))
    test_pairs = list(zip(test[src_col], test[dst_col]))
    if not test_pairs:
        return 0.0
    unseen = sum(1 for p in test_pairs if p not in train_pairs)
    return unseen / len(test_pairs)


def near_duplicate_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
    sample_size: int = 100_000,
    eps: float = 1e-6,
    seed: int = 0,
) -> float:
    """Estimate fraction of test flows within L2 distance `eps` of a train flow."""
    rng = np.random.default_rng(seed)
    n_test = min(sample_size, len(test))
    if n_test == 0 or len(train) == 0:
        return 0.0
    test_sample = test.sample(n=n_test, random_state=seed)[numeric_cols].to_numpy(dtype=float)
    train_arr = train[numeric_cols].to_numpy(dtype=float)
    # Subsample train too if huge, for tractable brute force on a laptop.
    if len(train_arr) > 20_000:
        idx = rng.choice(len(train_arr), size=20_000, replace=False)
        train_arr = train_arr[idx]

    hits = 0
    for row in test_sample:
        dists = np.linalg.norm(train_arr - row, axis=1)
        if dists.min() < eps:
            hits += 1
    return hits / n_test


def identity_leakage_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "canonical_label",
    src_col: str = "IPV4_SRC_ADDR",
    dst_col: str = "IPV4_DST_ADDR",
    src_port_col: str = "L4_SRC_PORT",
    dst_port_col: str = "L4_DST_PORT",
    seed: int = 0,
) -> dict[str, Any]:
    """Identity floor + unseen-pair rate + per-class attacker-host concentration.

    See docs/02_DATASETS.md §7.1.
    """

    def _port_bucket(p: pd.Series) -> pd.Series:
        return pd.cut(
            p, bins=[-1, 1023, 49151, 65535], labels=["well_known", "registered", "ephemeral"]
        ).astype(str)

    union_src = pd.concat([train[src_col], test[src_col]]).astype("category")
    union_dst = pd.concat([train[dst_col], test[dst_col]]).astype("category")

    def _encode_shared(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "src_ip": pd.Categorical(df[src_col], categories=union_src.cat.categories).codes,
                "dst_ip": pd.Categorical(df[dst_col], categories=union_dst.cat.categories).codes,
                "src_port_bucket": _port_bucket(df[src_port_col]).astype("category").cat.codes,
                "dst_port_bucket": _port_bucket(df[dst_port_col]).astype("category").cat.codes,
            }
        )

    x_train = _encode_shared(train)
    x_test = _encode_shared(test)
    y_train = train[label_col]
    y_test = test[label_col]

    clf = DecisionTreeClassifier(max_depth=8, random_state=seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    identity_floor = f1_score(y_test, pred, average="macro", zero_division=0)

    unseen_pair_rate = identity_overlap(train, test, src_col, dst_col)

    concentration = {}
    for c, part in train.groupby(label_col):
        if c == "benign":
            continue
        vc = part[src_col].value_counts()
        if len(vc) == 0:
            continue
        concentration[c] = {
            "n_distinct_src_ip": int(len(vc)),
            "top_ip_share": float(vc.iloc[0] / vc.sum()),
        }

    return {
        "identity_floor_macro_f1": float(identity_floor),
        "unseen_pair_rate": float(unseen_pair_rate),
        "per_class_host_concentration": concentration,
    }
