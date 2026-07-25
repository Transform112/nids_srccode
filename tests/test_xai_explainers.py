"""Tests for baseline explainers, explanation-quality metrics, and UNKNOWN triage."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from argus.config import load_config
from argus.models.argus import ArgusModel
from argus.models.epc import EPCHead
from argus.xai.explainers import (
    attention_weights_baseline,
    counterfactual_necessity,
    gnnexplainer_edge_mask,
    kernelshap_feature_importance,
    pgexplainer_edge_mask,
)
from argus.xai.metrics import feature_fidelity, necessity_rate, stability
from argus.xai.triage import (
    nearest_prototype_correspondence,
    render_triage_report,
    validate_unknown_clusters,
)


def _make_batch_dict(graph, device=torch.device("cpu")):
    node_feat = graph["node_feat"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    edge_dt = graph["edge_dt"].to(device)
    target_edge_idx = graph["target_edge_idx"].to(device)
    target_edge_index = edge_index[:, target_edge_idx]
    target_edge_attr = edge_attr[target_edge_idx]
    return {
        "node_feat": node_feat,
        "scale_short": (edge_index, edge_attr, edge_dt),
        "scale_mid": (edge_index, edge_attr, edge_dt),
        "scale_long": (edge_index, edge_attr, edge_dt),
        "target_edge_index": target_edge_index,
        "target_edge_attr": target_edge_attr,
        "target_labels": torch.zeros(target_edge_idx.shape[0], dtype=torch.long),
        "n_targets": target_edge_idx.shape[0],
    }


def _make_model(synthetic_graph, class_names):
    cfg = load_config()
    model = ArgusModel(cfg, f_e=synthetic_graph["f_e"], f_v=synthetic_graph["f_v"], class_names=class_names)
    model.eval()
    return model


def test_attention_weights_baseline_returns_normalised_weights(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    attn = attention_weights_baseline(model, batch, torch.device("cpu"))
    assert len(attn) > 0
    for node_id, weights in attn.items():
        assert torch.isfinite(weights).all()
        assert abs(float(weights.sum().item()) - 1.0) < 1e-4
    # Recording must be reset (opt-in side channel left clean afterwards).
    assert model.encoder.gnn_layers[-1].record_attention is False


def test_gnnexplainer_edge_mask_runs_and_produces_bounded_masks(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    result = gnnexplainer_edge_mask(model, batch, torch.device("cpu"), target_idx=0, class_idx=0, epochs=5)
    assert set(result.masks.keys()) == {"short", "mid", "long"}
    for m in result.masks.values():
        assert torch.isfinite(m).all()
        assert (m >= 0).all() and (m <= 1).all()
    assert np.isfinite(result.final_loss)


def test_pgexplainer_edge_mask_runs(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    result = pgexplainer_edge_mask(
        model, batch, torch.device("cpu"), target_idx=0, class_idx=0, f_e=synthetic_graph["f_e"], epochs=5,
    )
    for m in result.masks.values():
        assert torch.isfinite(m).all()
        assert (m >= 0).all() and (m <= 1).all()


def test_kernelshap_feature_importance_shape_and_finiteness(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    baseline = torch.zeros(synthetic_graph["f_e"])
    attribution = kernelshap_feature_importance(
        model, batch, torch.device("cpu"), target_idx=0, class_idx=0, baseline=baseline, n_samples=40, seed=0,
    )
    assert attribution.shape == (synthetic_graph["f_e"],)
    assert np.isfinite(attribution).all()


def test_counterfactual_necessity_runs_and_respects_budget(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    n_edges = synthetic_graph["edge_index"].shape[1]
    ranked = {"short": np.arange(n_edges), "mid": np.arange(n_edges), "long": np.arange(n_edges)}
    result = counterfactual_necessity(
        model, batch, torch.device("cpu"), target_idx=0, ranked_edges=ranked, max_removed_fraction=1.0,
    )
    assert 0.0 <= result.removed_fraction <= 1.0
    assert isinstance(result.flipped, bool)


def test_feature_fidelity_reasonable_values(synthetic_graph, class_names):
    torch.manual_seed(0)
    model = _make_model(synthetic_graph, class_names)
    batch = _make_batch_dict(synthetic_graph)
    rng = np.random.default_rng(0)
    attribution = rng.standard_normal(synthetic_graph["f_e"])
    baseline = torch.zeros(synthetic_graph["f_e"])
    result = feature_fidelity(
        model, batch, torch.device("cpu"), target_idx=0, class_idx=0,
        attribution=attribution, baseline=baseline, k=10,
    )
    assert np.isfinite(result.fidelity_plus)
    assert np.isfinite(result.fidelity_minus)
    assert result.k == 10
    assert abs(result.sparsity - (1.0 - 10 / synthetic_graph["f_e"])) < 1e-6


def test_necessity_rate_and_stability():
    assert necessity_rate([True, True, False, True]) == 0.75
    assert necessity_rate([]) == 0.0

    rng = np.random.default_rng(0)
    base = rng.standard_normal(20)
    identical = [base, base.copy(), base.copy()]
    assert stability(identical) > 0.99

    shuffled = [base, base[::-1].copy()]
    assert stability(shuffled) < stability(identical)


def test_render_triage_report_for_unknown_verdict():
    torch.manual_seed(0)
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    head = EPCHead(d_h=16, d_z=8, class_names=class_names)
    head.eval()
    h = torch.randn(2, 16)
    with torch.no_grad():
        outputs = head(h)
    report = render_triage_report(
        head, outputs, idx=0, class_names=class_names, gate={"short": 0.1, "mid": 0.2, "long": 0.7},
    )
    assert report.verdict == "UNKNOWN"
    assert len(report.nearest_prototypes) == 2
    assert "long time scale" in report.prose


def test_validate_unknown_clusters_kmeans_high_purity_on_separable_data():
    rng = np.random.default_rng(0)
    cluster_a = F.normalize(torch.tensor(rng.normal(loc=1.0, scale=0.05, size=(30, 8))), dim=-1).numpy()
    cluster_b = F.normalize(torch.tensor(rng.normal(loc=-1.0, scale=0.05, size=(30, 8))), dim=-1).numpy()
    embeddings = np.concatenate([cluster_a, cluster_b], axis=0)
    labels = np.array([0] * 30 + [1] * 30)
    result = validate_unknown_clusters(embeddings, labels, n_clusters=2, method="kmeans", seed=0)
    assert result.purity > 0.9
    assert result.nmi > 0.5
    assert result.ari > 0.5


def test_validate_unknown_clusters_empty_input():
    result = validate_unknown_clusters(np.zeros((0, 8)), np.zeros(0, dtype=int), n_clusters=2)
    assert result.purity == 0.0
    assert result.n_clusters == 0


def test_nearest_prototype_correspondence():
    torch.manual_seed(0)
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    head = EPCHead(d_h=16, d_z=8, class_names=class_names)
    bot_proto = head.prototype_bank.bank.data[
        torch.tensor([i == class_names.index("Bot") for i in head.prototype_bank.class_of])
    ][0]
    held_out_embeds = {"HeldOutBotVariant": bot_proto.unsqueeze(0).numpy()}
    correspondence = nearest_prototype_correspondence(head, held_out_embeds, class_names)
    assert correspondence["HeldOutBotVariant"] == "Bot"
