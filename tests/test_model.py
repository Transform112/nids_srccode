"""End-to-end ARGUS model tests."""

import pytest
import torch

from argus.config import load_config
from argus.models.argus import ArgusModel


def _make_scale_inputs(graph, device, k):
    """Duplicate edge tensors three times for S/M/L scales; node_feat is shared."""
    node_feat = graph["node_feat"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    edge_dt = graph["edge_dt"].to(device)
    target_edge_idx = graph["target_edge_idx"].to(device)
    target_edge_index = edge_index[:, target_edge_idx]
    target_edge_attr = edge_attr[target_edge_idx]
    return (
        node_feat,
        edge_index, edge_attr, edge_dt,
        edge_index, edge_attr, edge_dt,
        edge_index, edge_attr, edge_dt,
        target_edge_index, target_edge_attr,
    )


def test_model_forward_shapes(synthetic_graph, class_names):
    cfg = load_config()
    model = ArgusModel(
        cfg,
        f_e=synthetic_graph["f_e"],
        f_v=synthetic_graph["f_v"],
        class_names=class_names,
    )
    model.eval()
    inputs = _make_scale_inputs(synthetic_graph, torch.device("cpu"), cfg.graph.neighbour_cap)
    with torch.no_grad():
        out = model(*inputs)
    assert out["p_hat"].shape == (3, len(class_names))
    assert torch.isfinite(out["p_hat"]).all()


def test_gate_g0_overfit_capacity(synthetic_graph, class_names):
    """Gate G0: the model must be able to overfit a small fixed dataset.

    Each edge has a unique random feature vector and a deterministic label
    (index % num_classes), fixed across epochs. This is the simplest possible
    memorisation task and is the cheapest bug detector per docs/06_TRAINING.md §8.1.
    """
    torch.manual_seed(0)
    cfg = load_config(overrides=["model.layers=2", "graph.neighbour_cap=8"])
    model = ArgusModel(
        cfg,
        f_e=synthetic_graph["f_e"],
        f_v=synthetic_graph["f_v"],
        class_names=class_names,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    n_nodes = 8
    n_edges = 64
    node_feat = torch.randn(n_nodes, synthetic_graph["f_v"])
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(n_edges, synthetic_graph["f_e"])
    edge_dt = torch.rand(n_edges).abs()
    targets = torch.arange(n_edges) % len(class_names)
    target_edge_index = edge_index
    target_edge_attr = edge_attr

    for _epoch in range(80):
        opt.zero_grad()
        out = model(
            node_feat, edge_index, edge_attr, edge_dt,
            edge_index, edge_attr, edge_dt,
            edge_index, edge_attr, edge_dt,
            target_edge_index, target_edge_attr,
        )
        loss = torch.nn.functional.cross_entropy(out["logits"], targets)
        loss.backward()
        opt.step()
        if model.head.__class__.__name__ == "EPCHead":
            model.head.prototype_bank.post_step_normalize()

    model.eval()
    with torch.no_grad():
        out = model(
            node_feat, edge_index, edge_attr, edge_dt,
            edge_index, edge_attr, edge_dt,
            edge_index, edge_attr, edge_dt,
            target_edge_index, target_edge_attr,
        )
        pred = out["logits"].argmax(dim=1)
        acc = (pred == targets).float().mean()
    assert acc > 0.8, f"Gate G0 failed: train accuracy {acc.item():.3f} <= 0.8"


def test_no_batchnorm_in_model(synthetic_graph, class_names):
    cfg = load_config()
    model = ArgusModel(
        cfg,
        f_e=synthetic_graph["f_e"],
        f_v=synthetic_graph["f_v"],
        class_names=class_names,
    )
    for m in model.modules():
        assert "BatchNorm" not in m.__class__.__name__, "BatchNorm found in model"
