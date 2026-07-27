"""SRTEGLayer vectorisation equivalence (docs/BUGS.md #48).

`SRTEGLayer.forward` used to loop `for v in range(N)`, computing attention and
the robust aggregate one node at a time (~8.25 ms/node, ~3 h per CICIDS2018
epoch). It now runs one batched pass over the dense [N, K, ·] tensors.

These tests pin the refactor to the original semantics: `_reference_forward`
below is a faithful transcription of the removed loop, calling the *per-node*
aggregation helpers, and every case asserts the batched layer reproduces it.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from argus.models.aggregation import (
    mean_aggregate,
    soft_medoid,
    trimmed_mean,
    trimmed_std,
)
from argus.models.srteg import SRTEGLayer


def _reference_aggregate(layer, node_msgs, weights):
    """The per-node aggregation path exactly as RobustAggregator.forward did."""
    agg_kind = layer.aggregator.aggregation
    if agg_kind == "mean":
        return mean_aggregate(node_msgs, weights)
    if agg_kind == "trimmed":
        agg = trimmed_mean(node_msgs, weights, layer.aggregator.beta)
    elif agg_kind == "soft_medoid":
        agg = soft_medoid(node_msgs, weights, layer.aggregator.soft_medoid_temp)
    else:
        raise ValueError(agg_kind)
    multi = layer.aggregator.multi
    if multi is not None:
        a_spread = trimmed_std(node_msgs, weights, multi.beta)
        a_scale = math.log1p(node_msgs.shape[0]) / math.log1p(multi.k)
        combined = torch.cat([agg, a_spread, a_scale * agg], dim=0)
        agg = multi.project(combined)
    return agg


def _reference_forward(layer, x, neigh_attr, neigh_dt, time_enc, mask, batch=None):
    """Transcription of the original per-node loop, for equivalence checking."""
    n, k, _ = neigh_attr.shape
    src_state = x.unsqueeze(1).expand(-1, k, -1)
    msgs = layer.msg_mlp(torch.cat([src_state, neigh_attr, time_enc], dim=-1))

    out = torch.zeros_like(x)
    for v in range(n):
        m = mask[v]
        if not m.any():
            continue
        node_msgs = msgs[v, m]
        dt = neigh_dt[v, m] / layer.attn.scale_duration
        n_v = node_msgs.shape[0]
        if layer.te4_enabled:
            q = layer.attn.W_Q(x[v]).view(layer.attn.heads, layer.attn.d_k)
            kk = layer.attn.W_K(node_msgs).view(n_v, layer.attn.heads, layer.attn.d_k)
            scores = torch.einsum("hd,nhd->nh", q, kk) / math.sqrt(layer.attn.d_k)
            lambdas = F.softplus(layer.attn.lambda_hat)
            decay = -(lambdas.unsqueeze(0) * dt.unsqueeze(-1))
            attn = F.softmax(scores + decay, dim=0)
            weights = attn.mean(dim=1)
        else:
            weights = torch.full((n_v,), 1.0 / n_v, device=x.device, dtype=x.dtype)
        out[v] = _reference_aggregate(layer, node_msgs, weights)

    from argus.models.norm import GraphNorm

    concat = torch.cat([x, out], dim=-1)
    concat = layer.upd_norm(concat, batch) if isinstance(layer.upd_norm, GraphNorm) \
        else layer.upd_norm(concat)
    return x + layer.drop_path(layer.upd_mlp(concat))


def _make_layer(**kw):
    params = dict(
        d_h=16, d_t=8, heads=4, scale_duration=1.0, aggregation="trimmed",
        beta=0.2, soft_medoid_temp=1.0, multi_aggregator=True, k=32,
        dropout=0.0, droppath=0.0, norm_node="graphnorm", norm_mlp="layernorm",
        te4_enabled=True,
    )
    params.update(kw)
    layer = SRTEGLayer(**params)
    layer.eval()  # no dropout/droppath randomness
    return layer


def _make_inputs(n=9, k=6, d_h=16, d_t=8, seed=0):
    """Includes an isolated node (no neighbours), a single-neighbour node, and
    a fully-connected node — the branches the loop handled with `continue` and
    the `n_v - 2k < 1` trimming fallback."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_h, generator=g)
    neigh_attr = torch.randn(n, k, d_h, generator=g)
    neigh_dt = torch.rand(n, k, generator=g)
    time_enc = torch.randn(n, k, d_t, generator=g)

    mask = torch.rand(n, k, generator=g) > 0.4
    mask[0] = False                      # isolated node
    mask[1] = False; mask[1, 0] = True   # exactly one neighbour
    mask[2] = True                       # fully connected
    return x, neigh_attr, neigh_dt, time_enc, mask


@pytest.mark.parametrize("aggregation", ["trimmed", "mean", "soft_medoid"])
@pytest.mark.parametrize("te4_enabled", [True, False])
def test_batched_matches_loop(aggregation, te4_enabled):
    layer = _make_layer(aggregation=aggregation, te4_enabled=te4_enabled)
    x, na, ndt, te, mask = _make_inputs()

    got = layer(x, na, ndt, te, mask)
    want = _reference_forward(layer, x, na, ndt, te, mask)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("aggregation", ["trimmed", "mean", "soft_medoid"])
@pytest.mark.parametrize("te4_enabled", [True, False])
def test_batched_gradients_match_the_loop(aggregation, te4_enabled):
    """Equal forward output is not enough. The batched path reaches the same
    numbers through `masked_fill` sentinels, a shared `argsort`, and `gather` —
    any of which could route gradient differently while the forward value
    matches, degrading training silently. G0 dropped 0.9915 -> 0.9360 across
    the window that contains this rewrite, so the backward pass needs the same
    equivalence check the forward already had (docs/BUGS.md #48, #54).
    """
    layer = _make_layer(aggregation=aggregation, te4_enabled=te4_enabled)
    x, na, ndt, te, mask = _make_inputs()

    def grads(fn):
        layer.zero_grad(set_to_none=True)
        xg = x.clone().requires_grad_(True)
        nag = na.clone().requires_grad_(True)
        fn(xg, nag).pow(2).sum().backward()
        return (xg.grad, nag.grad,
                {n: (p.grad.clone() if p.grad is not None else None)
                 for n, p in layer.named_parameters()})

    gx_b, gna_b, gp_b = grads(lambda a, b: layer(a, b, ndt, te, mask))
    gx_r, gna_r, gp_r = grads(lambda a, b: _reference_forward(layer, a, b, ndt, te, mask))

    torch.testing.assert_close(gx_b, gx_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(gna_b, gna_r, rtol=1e-4, atol=1e-5)
    for name in gp_r:
        if gp_r[name] is None and gp_b[name] is None:
            continue
        assert (gp_b[name] is None) == (gp_r[name] is None), f"{name}: gradient presence differs"
        torch.testing.assert_close(gp_b[name], gp_r[name], rtol=1e-4, atol=1e-5,
                                   msg=lambda m, n=name: f"param {n}: {m}")


def test_no_masked_sentinel_leaks_into_gradients():
    """`_sorted_keep` pads masked slots with `finfo.max` before sorting. If a
    padded slot were ever kept, the forward would blow up — but a subtler
    failure is a finite forward with an enormous or NaN gradient."""
    layer = _make_layer()
    x, na, ndt, te, mask = _make_inputs()
    na = na.clone().requires_grad_(True)
    layer(x, na, ndt, te, mask).pow(2).sum().backward()
    assert torch.isfinite(na.grad).all()
    assert float(na.grad.abs().max()) < 1e4, "sentinel-scale magnitude leaked into the gradient"
    # Masked slots contribute nothing, so their gradient must be exactly zero.
    assert float(na.grad[~mask].abs().max()) == 0.0


def test_isolated_node_row_is_untouched_residual():
    """A node with no neighbours must aggregate to exactly zero, as the loop's
    `continue` left `out[v]` at its zero initialisation."""
    layer = _make_layer()
    x, na, ndt, te, mask = _make_inputs()
    n, k = mask.shape
    agg = layer.aggregator.forward_batched(
        torch.randn(n, k, 16), torch.rand(n, k), mask
    )
    assert torch.count_nonzero(agg[0]) == 0, "isolated node must aggregate to zero"
    assert torch.count_nonzero(agg[2]) > 0, "connected node must aggregate to non-zero"


def test_gradients_flow_to_inputs_and_params():
    layer = _make_layer()
    x, na, ndt, te, mask = _make_inputs()
    x = x.clone().requires_grad_(True)
    out = layer(x, na, ndt, te, mask)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # lambda_hat drives the time decay; it must still receive gradient.
    assert layer.attn.lambda_hat.grad is not None
    assert torch.isfinite(layer.attn.lambda_hat.grad).all()


def test_no_nan_when_all_nodes_isolated():
    """All-masked rows must not produce NaN (the softmax sentinel guard)."""
    layer = _make_layer()
    x, na, ndt, te, mask = _make_inputs()
    mask = torch.zeros_like(mask)
    out = layer(x, na, ndt, te, mask)
    assert torch.isfinite(out).all()
