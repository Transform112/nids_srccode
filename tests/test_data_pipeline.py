"""Data pipeline tests: clean, canonical, subsample, splits, audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.data.canonical import canonicalise_labels
from argus.data.clean import clean
from argus.data.splits import protocol_a_split, protocol_b_split, sample_holdout_sets
from argus.data.audit import audit_split, identity_leakage_audit
from argus.data.subsample import stratified_temporal_subsample
from test_features_pipeline import _make_synthetic_raw


def test_clean_drops_nulls_and_dedups():
    df = _make_synthetic_raw(n=200)
    df.loc[0, "Attack"] = None
    dup = df.iloc[[1]]
    df = pd.concat([df, dup], ignore_index=True)
    cleaned, report = clean(df)
    assert report["dropped_null"] == 1
    assert report["dropped_duplicates"] >= 1
    assert cleaned["FLOW_START_MILLISECONDS"].dtype == np.int64


def test_canonicalise_labels_cicids():
    df = _make_synthetic_raw(n=50)
    df["Attack"] = "Benign"
    out = canonicalise_labels(df, "cicids2018")
    assert (out["canonical_label"] == "benign").all()


def test_canonicalise_raises_on_unknown_label():
    df = _make_synthetic_raw(n=5)
    df["Attack"] = "TotallyUnknownLabel"
    with pytest.raises(ValueError):
        canonicalise_labels(df, "cicids2018")


def test_protocol_a_split_every_class_present():
    df = _make_synthetic_raw(n=600)
    df = canonicalise_labels(df, "cicids2018")
    splits = protocol_a_split(df, label_col="canonical_label")
    train_classes = set(splits["train"]["canonical_label"].unique())
    test_classes = set(splits["test"]["canonical_label"].unique())
    assert test_classes.issubset(train_classes)


def test_protocol_b_split_holdout_labelled_unknown():
    df = _make_synthetic_raw(n=600)
    df = canonicalise_labels(df, "cicids2018")
    splits = protocol_b_split(df, holdout_classes=["bot"], label_col="canonical_label")
    assert "bot" not in splits["train"]["canonical_label"].unique()
    assert "UNKNOWN" in splits["test"]["canonical_label"].unique()


def test_sample_holdout_sets_distinct():
    holdouts = sample_holdout_sets(["a", "b", "c", "d", "e"], holdout_size=2, repeats=3, seed=1)
    assert len(holdouts) <= 3
    assert len(set(tuple(h) for h in holdouts)) == len(holdouts)


def test_audit_split_passes_and_raises_on_duplicate():
    df = _make_synthetic_raw(n=600)
    df = canonicalise_labels(df, "cicids2018")
    splits = protocol_a_split(df, label_col="canonical_label")
    report = audit_split(splits, protocol="A", label_col="canonical_label")
    assert report["duplicate_check"] == "passed"

    # Force a duplicate across train/test to verify it raises.
    bad_splits = dict(splits)
    bad_splits["test"] = pd.concat([splits["test"], splits["train"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        audit_split(bad_splits, protocol="A", label_col="canonical_label")


def test_stratified_subsample_keeps_minority_full():
    df = _make_synthetic_raw(n=1000)
    df = canonicalise_labels(df, "cicids2018")
    out, report = stratified_temporal_subsample(
        df, target_total=200, minority_threshold=100, benign_floor_fraction=0.5
    )
    minority_classes = [c for c, n in report["pre_counts"].items() if n <= 100]
    for c in minority_classes:
        assert report["post_counts"].get(c, 0) == report["pre_counts"][c]


def test_identity_leakage_audit_runs():
    df = _make_synthetic_raw(n=400)
    df = canonicalise_labels(df, "cicids2018")
    splits = protocol_a_split(df, label_col="canonical_label")
    result = identity_leakage_audit(splits["train"], splits["test"], label_col="canonical_label")
    assert 0.0 <= result["identity_floor_macro_f1"] <= 1.0
    assert 0.0 <= result["unseen_pair_rate"] <= 1.0
