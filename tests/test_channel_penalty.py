"""Channel gradient penalty tests (C2 mechanism)."""

from __future__ import annotations

import torch

from argus.losses.channel_penalty import ChannelPenaltyLoss


def test_channel_penalty_zero_when_balanced():
    torch.manual_seed(0)
    edge_attr = torch.randn(8, 10, requires_grad=True)
    a_idx = torch.tensor([0, 1, 2, 3, 4])
    b_idx = torch.tensor([5, 6, 7, 8, 9])
    # Loss depends equally on both channels -> ratio ~ 0.5, below default rho=0.5
    loss = (edge_attr[:, a_idx] ** 2).sum() + (edge_attr[:, b_idx] ** 2).sum()
    penalty_fn = ChannelPenaltyLoss(rho=0.5)
    penalty = penalty_fn(loss, edge_attr, a_idx, b_idx)
    assert penalty.item() >= 0.0


def test_channel_penalty_positive_when_channel_a_dominates():
    torch.manual_seed(0)
    edge_attr = torch.randn(8, 10, requires_grad=True)
    a_idx = torch.tensor([0, 1, 2, 3, 4])
    b_idx = torch.tensor([5, 6, 7, 8, 9])
    # Loss depends heavily on channel A only -> ratio ~ 1.0, above rho=0.3
    loss = (edge_attr[:, a_idx] ** 2).sum() * 100.0 + (edge_attr[:, b_idx] ** 2).sum() * 0.001
    penalty_fn = ChannelPenaltyLoss(rho=0.3)
    penalty = penalty_fn(loss, edge_attr, a_idx, b_idx)
    assert penalty.item() > 0.0


def test_channel_penalty_gradient_flows_to_params():
    """The penalty must be differentiable w.r.t. upstream parameters (double backward)."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(10, 10))
    edge_attr_leaf = torch.randn(8, 10)
    edge_attr = edge_attr_leaf @ w
    edge_attr.retain_grad()
    a_idx = torch.tensor([0, 1, 2, 3, 4])
    b_idx = torch.tensor([5, 6, 7, 8, 9])
    loss = (edge_attr[:, a_idx] ** 2).sum() + (edge_attr[:, b_idx] ** 2).sum() * 0.01
    penalty_fn = ChannelPenaltyLoss(rho=0.3)
    penalty = penalty_fn(loss, edge_attr, a_idx, b_idx)
    total = loss + penalty
    total.backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
