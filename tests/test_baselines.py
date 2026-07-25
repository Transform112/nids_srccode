"""Baseline model tests: tabular, identity-only, E-GraphSAGE, EGATv2,
Anomal-E, and post-hoc OSR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from argus.models.baselines.anomal_e import AnomalE
from argus.models.baselines.egatv2 import EGATv2
from argus.models.baselines.egraphsage import EGraphSAGE
from argus.models.baselines.identity_only import IdentityOnlyClassifier
from argus.models.baselines.posthoc_osr import (
    OpenMax,
    PostHocOSRBaseline,
    energy_score,
    odin_score,
)
from argus.models.baselines.tabular import TabularBaseline
from argus.eval.metrics import closed_set_report, per_tier_macro_f1


def test_tabular_extra_trees_fit_predict():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 10))
    y = rng.integers(0, 3, 200)
    baseline = TabularBaseline.extra_trees(n_estimators=10, seed=0)
    baseline.fit(x, y)
    pred = baseline.predict(x)
    assert pred.shape == y.shape
    proba = baseline.predict_proba(x)
    assert proba.shape == (200, 3)


def test_tabular_mlp_fit_predict():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, 100)
    baseline = TabularBaseline.mlp(hidden_sizes=(8,), seed=0)
    baseline.fit(x, y)
    pred = baseline.predict(x)
    assert pred.shape == y.shape


def test_identity_only_classifier():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "IPV4_SRC_ADDR": rng.integers(0, 10, n).astype(str),
        "IPV4_DST_ADDR": rng.integers(0, 10, n).astype(str),
        "L4_SRC_PORT": rng.integers(1, 65535, n),
        "L4_DST_PORT": rng.choice([22, 80, 443], n),
    })
    y = rng.integers(0, 2, n)
    clf = IdentityOnlyClassifier(max_depth=4, seed=0)
    clf.fit(df, y)
    pred = clf.predict(df)
    assert pred.shape == y.shape


def test_egraphsage_forward():
    torch.manual_seed(0)
    f_e, d_h, num_classes = 20, 16, 3
    model = EGraphSAGE(f_e=f_e, d_h=d_h, num_classes=num_classes, layers=2)
    n_nodes = 6
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, f_e)
    target_edge_index = edge_index[:, :2]
    target_edge_attr = edge_attr[:2]
    logits = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    assert logits.shape == (2, num_classes)
    assert torch.isfinite(logits).all()


def test_egatv2_forward():
    torch.manual_seed(0)
    f_e, d_h, num_classes = 20, 16, 3
    model = EGATv2(f_e=f_e, d_h=d_h, num_classes=num_classes, layers=2, heads=4)
    n_nodes = 6
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, f_e)
    target_edge_index = edge_index[:, :2]
    target_edge_attr = edge_attr[:2]
    logits = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    assert logits.shape == (2, num_classes)
    assert torch.isfinite(logits).all()


def test_egatv2_backward_produces_gradients():
    torch.manual_seed(0)
    f_e, d_h, num_classes = 8, 16, 3
    model = EGATv2(f_e=f_e, d_h=d_h, num_classes=num_classes, layers=2, heads=4)
    n_nodes = 6
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, f_e)
    target_edge_index = edge_index[:, :2]
    target_edge_attr = edge_attr[:2]
    logits = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_egatv2_handles_node_with_no_incoming_edges():
    """A node absent from `dst` must not produce NaNs from the softmax max-shift."""
    torch.manual_seed(0)
    f_e, d_h, num_classes = 8, 16, 2
    model = EGATv2(f_e=f_e, d_h=d_h, num_classes=num_classes, layers=1, heads=2)
    n_nodes = 4  # node 3 has no incoming edge
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.randn(2, f_e)
    target_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    target_edge_attr = edge_attr[:1]
    logits = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    assert torch.isfinite(logits).all()


def test_closed_set_report_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    class_names = ["a", "b", "c"]
    report = closed_set_report(y_true, y_pred, class_names)
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert set(report["per_class_f1"].keys()) == set(class_names)
    tiers = per_tier_macro_f1(report["per_class_f1"], report["support"], tail_threshold=0)
    assert "extreme" in tiers


def test_anomal_e_forward():
    torch.manual_seed(0)
    f_e, d_h, latent_dim = 20, 16, 8
    model = AnomalE(f_e=f_e, d_h=d_h, latent_dim=latent_dim)
    n_nodes = 6
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.randn(5, f_e)
    target_edge_index = edge_index[:, :3]
    target_edge_attr = edge_attr[:3]
    out = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    assert "anomaly_score" in out
    assert out["anomaly_score"].shape == (3,)
    assert torch.isfinite(out["anomaly_score"]).all()
    assert out["anomaly_score"].min() >= 0  # MSE is non-negative


def test_anomal_e_backward():
    torch.manual_seed(0)
    f_e, d_h, latent_dim = 8, 16, 4
    model = AnomalE(f_e=f_e, d_h=d_h, latent_dim=latent_dim)
    n_nodes = 4
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, f_e)
    target_edge_index = edge_index[:, :2]
    target_edge_attr = edge_attr[:2]
    out = model(n_nodes, edge_index, edge_attr, target_edge_index, target_edge_attr)
    loss = out["anomaly_score"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_energy_score_is_finite():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((100, 5))
    scores = energy_score(logits)
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_odin_score_range():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((100, 5))
    scores = odin_score(logits)
    assert scores.shape == (100,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_openmax_fit_score():
    rng = np.random.default_rng(0)
    n_classes = 4
    # Generate well-separated logits
    means = np.eye(n_classes) * 5
    logits = np.vstack([
        means[i] + rng.standard_normal((50, n_classes)) * 0.5 for i in range(n_classes)
    ])
    labels = np.repeat(np.arange(n_classes), 50)
    om = OpenMax(tailsize=10)
    om.fit(logits, labels)
    scores = om.score(logits)
    assert scores.shape == (200,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_posthoc_osr_baseline_energy():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((100, 5))
    labels = rng.integers(0, 5, 100)
    for method in ["energy", "odin"]:
        baseline = PostHocOSRBaseline(method)
        baseline.fit(logits, labels)
        scores = baseline.score(logits)
        assert scores.shape == (100,)
        assert np.isfinite(scores).all()


def test_posthoc_osr_baseline_openmax():
    rng = np.random.default_rng(0)
    n_classes = 4
    means = np.eye(n_classes) * 5
    logits = np.vstack([
        means[i] + rng.standard_normal((30, n_classes)) * 0.5 for i in range(n_classes)
    ])
    labels = np.repeat(np.arange(n_classes), 30)
    baseline = PostHocOSRBaseline("openmax", tailsize=10)
    baseline.fit(logits, labels)
    scores = baseline.score(logits)
    assert scores.shape == (120,)
    assert (scores >= 0).all() and (scores <= 1).all()
