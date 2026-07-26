"""Class-imbalance mechanisms (docs/06_TRAINING.md §4, docs/BUGS.md #50-#53).

CICIDS2018's Protocol-A train split spans 34,014:1 — `ddos_hoic` at 476,204
rows against `brute_xss` at 14. The design specifies effective-number loss
weights, class-balanced target subsampling, and a minimum-count guard; none
were wired up, and model selection ran on accuracy, which on this split is
almost entirely a measure of how well the two largest classes are fit.
"""

from __future__ import annotations

import math

import pytest
import torch

import numpy as np

from argus.losses.class_balance import (
    balanced_target_indices,
    effective_number_weights,
    effective_target_counts,
    min_count_excluded_classes,
)
from argus.train.gates import gate_g8_tail_collapse
from argus.train.loop import EpochResult

# The real Protocol-A training histogram (scripts/02 output).
CICIDS_TRAIN = {
    "ddos_hoic": 476204, "benign": 463428, "brute_ftp": 185850, "brute_ssh": 131931,
    "infiltration": 131706, "bot": 89205, "dos_slowhttptest": 73885, "dos_hulk": 70053,
    "ddos_loic_http": 46935, "dos_goldeneye": 42910, "dos_slowloris": 25228,
    "ddos_loic_udp": 2415, "brute_web": 1132, "sql_injection": 308, "brute_xss": 14,
}
NAMES = list(CICIDS_TRAIN)


# --- effective-number weights ------------------------------------------------

def test_weights_match_the_closed_form():
    nu = 0.999
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=nu)
    raw = [(1 - nu) / (1 - nu ** n) for n in CICIDS_TRAIN.values()]
    scale = len(raw) / sum(raw)
    for got, want in zip(w.tolist(), raw):
        assert got == pytest.approx(want * scale, rel=1e-5)


def test_weights_average_to_one():
    """Rescaling keeps the loss on the same scale as unweighted, so the tuned
    stage1_lr and grad_clip remain valid."""
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.999)
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-6)


def test_saturation_compresses_the_ratio():
    """The whole point of the effective-number form over inverse frequency:
    a 34,014:1 frequency ratio becomes a ~72:1 weight ratio. Inverse frequency
    would hand fourteen brute_xss rows more gradient than all of ddos_hoic."""
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.999)
    freq_ratio = max(CICIDS_TRAIN.values()) / min(CICIDS_TRAIN.values())
    weight_ratio = float(w.max() / w.min())
    assert freq_ratio > 30_000
    assert weight_ratio == pytest.approx(71.9, abs=0.5)


def test_no_underflow_on_large_counts():
    """nu ** 476204 is ~1e-207 — it must not produce a NaN or a zero weight."""
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.999)
    assert torch.isfinite(w).all()
    assert float(w.min()) > 0.0


def test_rarer_class_never_gets_less_weight():
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.999).tolist()
    counts = list(CICIDS_TRAIN.values())
    for (n_a, w_a), (n_b, w_b) in zip(zip(counts, w), zip(counts[1:], w[1:])):
        if n_a > n_b:
            assert w_a <= w_b + 1e-9


def test_absent_class_gets_zero_weight():
    """Normalisation averages over *present* classes, so adding a class that
    never appears in training does not deflate everyone else's weight."""
    w = effective_number_weights(["a", "b", "ghost"], {"a": 10, "b": 20}, nu=0.999)
    assert float(w[2]) == 0.0
    assert float(w[:2].mean()) == pytest.approx(1.0, rel=1e-6)


def test_nu_zero_is_the_unweighted_case():
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.0)
    assert w.tolist() == pytest.approx([1.0] * len(NAMES))


def test_rejects_nu_out_of_range():
    with pytest.raises(ValueError):
        effective_number_weights(NAMES, CICIDS_TRAIN, nu=1.0)


def test_weights_are_usable_by_cross_entropy():
    """The tensor must plug straight into F.cross_entropy(weight=...)."""
    w = effective_number_weights(NAMES, CICIDS_TRAIN, nu=0.999)
    logits = torch.randn(8, len(NAMES))
    targets = torch.randint(0, len(NAMES), (8,))
    loss = torch.nn.functional.cross_entropy(logits, targets, weight=w)
    assert torch.isfinite(loss)


# --- minimum-count guard -----------------------------------------------------

def test_min_count_guard_selects_only_brute_xss():
    """At min_count=100 exactly one CICIDS2018 class qualifies: brute_xss (14).
    sql_injection at 308 stays in Stage-1 training."""
    excluded = min_count_excluded_classes(NAMES, CICIDS_TRAIN, 100)
    assert [NAMES[i] for i in excluded] == ["brute_xss"]


def test_min_count_guard_ignores_absent_classes():
    """A zero-count class is handled by its zero weight; flagging it here would
    also drop it from the macro-F1 denominator, hiding a real failure."""
    assert min_count_excluded_classes(["a", "ghost"], {"a": 5000}, 100) == []


def test_min_count_guard_disabled_at_zero():
    assert min_count_excluded_classes(NAMES, CICIDS_TRAIN, 0) == []


# --- class-balanced target subsampling ---------------------------------------

def test_caps_the_head_without_touching_the_tail():
    targets = torch.tensor([0] * 500 + [1] * 40 + [2] * 3)
    idx = balanced_target_indices(targets, n_per_class=32)
    counts = torch.bincount(targets[idx], minlength=3).tolist()
    assert counts == [32, 32, 3], "cap the head, keep the tail whole"


def test_never_duplicates_a_row():
    """Explicitly not oversampling: at n_c=14 duplication memorises
    (docs/06_TRAINING.md §4)."""
    targets = torch.tensor([0] * 100 + [1] * 2)
    idx = balanced_target_indices(targets, n_per_class=32)
    assert len(set(idx.tolist())) == len(idx)


def test_indices_stay_sorted():
    targets = torch.randint(0, 4, (200,))
    idx = balanced_target_indices(targets, n_per_class=8)
    assert idx.tolist() == sorted(idx.tolist())


def test_excluded_classes_are_dropped_entirely():
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    idx = balanced_target_indices(targets, n_per_class=32, exclude_class_ids=[1])
    assert 1 not in targets[idx].tolist()
    assert sorted(targets[idx].tolist()) == [0, 0, 2, 2]


def test_disabled_cap_returns_everything():
    targets = torch.randint(0, 4, (50,))
    idx = balanced_target_indices(targets, n_per_class=None)
    assert idx.tolist() == list(range(50))


def test_all_excluded_returns_empty():
    """The loss closure must handle this — a bin can be entirely one class."""
    targets = torch.tensor([3, 3, 3])
    assert balanced_target_indices(targets, 32, exclude_class_ids=[3]).numel() == 0


def test_sampling_varies_with_the_rng():
    targets = torch.tensor([0] * 200)
    torch.manual_seed(0)
    a = balanced_target_indices(targets, n_per_class=32)
    torch.manual_seed(1)
    b = balanced_target_indices(targets, n_per_class=32)
    assert a.tolist() != b.tolist()


# --- post-cap effective counts -----------------------------------------------

def test_effective_counts_apply_the_cap_per_bin():
    # class 0: 40 flows in bin 0 -> capped to 4; class 1: 2 flows -> kept.
    labels = np.array([0] * 40 + [1] * 2)
    bins = np.zeros(42, dtype=np.int64)
    assert effective_target_counts(labels, bins, 2, n_per_class=4) == [4, 2]


def test_effective_counts_accumulate_across_bins():
    """The same 40 flows spread over 10 bins are barely capped at all — this is
    exactly why `infiltration` gains share while `ddos_hoic` loses it."""
    labels = np.zeros(40, dtype=np.int64)
    spread = np.repeat(np.arange(10), 4)
    burst = np.zeros(40, dtype=np.int64)
    assert effective_target_counts(labels, spread, 1, n_per_class=4) == [40]
    assert effective_target_counts(labels, burst, 1, n_per_class=4) == [4]


def test_effective_counts_without_a_cap_are_raw_counts():
    labels = np.array([0, 0, 1, 2, 2, 2])
    bins = np.array([0, 0, 0, 1, 1, 1])
    assert effective_target_counts(labels, bins, 3, n_per_class=None) == [2, 1, 3]


def test_effective_counts_cover_absent_classes():
    labels = np.array([0, 0])
    assert effective_target_counts(labels, np.zeros(2, dtype=np.int64), 4, 8) == [2, 0, 0, 0]


def test_capping_reorders_which_class_dominates():
    """The finding that makes raw-count weighting wrong: after capping, the
    burst class is no longer the biggest contributor."""
    labels = np.array([0] * 1000 + [1] * 100)
    bins = np.concatenate([np.zeros(1000, dtype=np.int64), np.arange(100)])
    raw = effective_target_counts(labels, bins, 2, n_per_class=None)
    capped = effective_target_counts(labels, bins, 2, n_per_class=32)
    assert raw[0] > raw[1]
    assert capped[0] < capped[1]


# --- macro-F1 as the selection metric ----------------------------------------

def _confusion_for(preds_by_true: dict[int, dict[int, int]], c: int) -> torch.Tensor:
    cm = torch.zeros(c, c, dtype=torch.long)
    for t, row in preds_by_true.items():
        for p, n in row.items():
            cm[t, p] = n
    return cm


def test_macro_f1_exposes_what_accuracy_hides():
    """The regression that cost three GPU runs: a model predicting only the two
    largest classes. Accuracy calls that 0.54; macro-F1 calls it 0.10."""
    c = 15
    counts = list(CICIDS_TRAIN.values())
    rows = {}
    for i, n in enumerate(counts):
        rows[i] = {i: n} if i < 2 else {0: n}  # everything else -> ddos_hoic
    cm = _confusion_for(rows, c)
    result = EpochResult(loss=0.0, n_batches=1, n_targets=int(cm.sum()),
                         correct=int(cm.diagonal().sum()), confusion=cm)
    assert result.accuracy == pytest.approx(0.54, abs=0.01)
    assert result.macro_f1 == pytest.approx(0.103, abs=0.005)
    assert result.accuracy > 5 * result.macro_f1


def test_macro_f1_is_one_for_a_perfect_classifier():
    cm = torch.diag(torch.tensor([100, 50, 3]))
    r = EpochResult(0.0, 1, 153, 153, confusion=cm)
    assert r.macro_f1 == pytest.approx(1.0)


def test_macro_f1_ignores_classes_with_no_support():
    """A rare class absent from a pass must not be scored 0 — otherwise the
    metric depends on how the tail happened to fall across bins."""
    cm = torch.zeros(4, 4, dtype=torch.long)
    cm[0, 0] = 100
    cm[1, 1] = 50
    r = EpochResult(0.0, 1, 150, 150, confusion=cm)
    assert set(r.per_class_f1) == {0, 1}
    assert r.macro_f1 == pytest.approx(1.0)


def test_macro_f1_excludes_min_count_classes():
    """brute_xss is deliberately untrained; scoring it would fail G8 by design."""
    cm = torch.diag(torch.tensor([100, 50, 10]))
    cm[2, 2] = 0
    cm[2, 0] = 10  # class 2 always mispredicted
    r = EpochResult(0.0, 1, 160, 150, confusion=cm, metric_ignore_class_ids=(2,))
    assert 2 not in r.per_class_f1
    assert r.macro_f1 > 0.9


def test_macro_f1_penalises_a_single_dead_class():
    cm = torch.diag(torch.tensor([100, 100, 100, 100]))
    cm[3, 3] = 0
    cm[3, 0] = 100
    r = EpochResult(0.0, 1, 400, 300, confusion=cm)
    assert r.per_class_f1[3] == 0.0
    assert r.macro_f1 < 0.75


def test_no_confusion_means_no_macro_f1():
    """Two-element loss returns stay supported; they just get no F1."""
    r = EpochResult(0.0, 1, 10, 5)
    assert r.per_class_f1 == {}
    assert r.macro_f1 == 0.0


# --- G8 tail collapse --------------------------------------------------------

def test_g8_fails_on_a_collapsed_class():
    ok, msg = gate_g8_tail_collapse({"benign": 0.99, "bot": 0.9, "brute_web": 0.0})
    assert not ok
    assert "brute_web" in msg


def test_g8_passes_when_every_class_scores():
    ok, _ = gate_g8_tail_collapse({"benign": 0.99, "bot": 0.9, "brute_web": 0.01})
    assert ok


def test_g8_catches_the_bug_49_shape():
    """A near-constant `bot` predictor: one class fine, everything else zero."""
    per_class = {n: (0.87 if n == "bot" else 0.0) for n in NAMES}
    ok, msg = gate_g8_tail_collapse(per_class)
    assert not ok
    assert "14/15" in msg


def test_g8_tolerance_is_configurable():
    ok, _ = gate_g8_tail_collapse({"a": 0.9, "b": 0.0}, max_collapsed=1)
    assert ok


def test_g8_and_g0_disagree_on_the_same_model():
    """The reason G8 exists. G0 trains on a class-balanced 1,000-flow subset,
    so a tail-blind model still memorises it and passes at 0.99."""
    from argus.train.gates import gate_g0_capacity
    g0_ok, _ = gate_g0_capacity(0.9947)
    g8_ok, _ = gate_g8_tail_collapse({n: (0.9 if i < 2 else 0.0)
                                      for i, n in enumerate(NAMES)})
    assert g0_ok and not g8_ok


# --- sanity on the published histogram ---------------------------------------

def test_the_split_itself_is_proportional():
    """Not a code test — a guard on the premise. protocol_a_split is per-class
    stratified, so imbalance is inherited from the data, never introduced by
    the split. If this drifts, the imbalance story changes."""
    from argus.data.splits import protocol_a_split
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 20_000
    labels = rng.choice(["big", "mid", "tiny"], size=n, p=[0.9, 0.099, 0.001])
    df = pd.DataFrame({
        "canonical_label": labels,
        "FLOW_START_MILLISECONDS": np.sort(rng.integers(0, 10**9, size=n)),
    })
    splits = protocol_a_split(df)
    base = df["canonical_label"].value_counts(normalize=True)
    for name in ("train", "val", "test"):
        got = splits[name]["canonical_label"].value_counts(normalize=True)
        for cls in base.index:
            assert math.isclose(got.get(cls, 0.0), base[cls], abs_tol=0.005), \
                f"{name} split is not proportional for {cls}"
