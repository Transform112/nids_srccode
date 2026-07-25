"""Shared test fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_graph(device):
    """Return a tiny graph for model tests.

    4 nodes, ~10 edges, F_e=147, F_v=18.
    """
    torch.manual_seed(42)
    n_nodes = 4
    f_e = 147
    f_v = 18
    k = 4

    node_feat = torch.randn(n_nodes, f_v, device=device)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2, 1, 3, 0, 1], [1, 2, 3, 0, 2, 3, 0, 2, 3, 1]],
        dtype=torch.long,
        device=device,
    )
    edge_attr = torch.randn(edge_index.shape[1], f_e, device=device)
    edge_dt = torch.rand(edge_index.shape[1], device=device).abs()
    target_edge_idx = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    return {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_dt": edge_dt,
        "target_edge_idx": target_edge_idx,
        "f_e": f_e,
        "f_v": f_v,
        "k": k,
    }


@pytest.fixture
def class_names():
    return ["Benign", "FTP-BruteForce", "Bot"]
