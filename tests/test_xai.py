"""Native evidence attribution tests (C4 evidence base)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from argus.xai.evidence_attrib import (
    embedding_attribution,
    integrated_gradients_feature_attribution,
    verify_embedding_decomposition,
)
from argus.models.epc import EPCHead


def test_embedding_decomposition_is_exact():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(5, 16), dim=-1)
    prototype = F.normalize(torch.randn(16), dim=-1)
    tau = torch.tensor(0.1)
    margin = 0.35
    attributed, actual = verify_embedding_decomposition(z, prototype, tau, margin)
    torch.testing.assert_close(attributed, actual, atol=1e-5, rtol=1e-4)


def test_embedding_attribution_shape():
    z = F.normalize(torch.randn(3, 8), dim=-1)
    prototype = F.normalize(torch.randn(8), dim=-1)
    attr = embedding_attribution(z, prototype, tau=0.2)
    assert attr.shape == (3, 8)


def test_integrated_gradients_completeness_on_epc_head():
    """IG attribution must sum to f(x) - f(baseline) within tolerance (completeness axiom)."""
    torch.manual_seed(0)
    class_names = ["Benign", "FTP-BruteForce", "Bot"]
    head = EPCHead(
        d_h=16, d_z=8, class_names=class_names,
        class_counts={"Benign": 100_000, "FTP-BruteForce": 10_000, "Bot": 5_000},
        sub_prototypes_benign=2, sub_prototypes_attack_large=1, sub_prototypes_attack_small=1,
        fp32_head=False,
    )
    head.eval()

    def forward_log_e(h: torch.Tensor) -> torch.Tensor:
        out = head(h)
        return out["log_e"]

    x = torch.randn(16)
    baseline = torch.zeros(16)
    attribution, f_x, f_base = integrated_gradients_feature_attribution(
        forward_log_e, x, baseline, class_idx=0, steps=100,
    )
    completeness_gap = abs(attribution.sum().item() - (f_x - f_base))
    # IG completeness holds exactly for piecewise-linear/smooth functions with
    # enough steps; allow a small numerical tolerance.
    assert completeness_gap < 0.05, f"IG completeness violated: gap={completeness_gap}"
