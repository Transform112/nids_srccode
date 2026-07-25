"""EPC head tests."""

import pytest
import torch

from argus.models.epc import EPCHead


@pytest.fixture
def epc_head(class_names):
    return EPCHead(
        d_h=32,
        d_z=16,
        class_names=class_names,
        class_counts={"Benign": 100_000, "FTP-BruteForce": 10_000, "Bot": 5_000},
        sub_prototypes_benign=2,
        sub_prototypes_attack_large=2,
        sub_prototypes_attack_small=1,
        margin_m=0.35,
        tau_start=1.0,
        tau_final=0.10,
        tau_min=0.02,
        log_evidence_clamp=15.0,
        fp32_head=False,
    )


def test_epc_forward_shapes(epc_head):
    h = torch.randn(5, 32)
    out = epc_head(h)
    assert out["p_hat"].shape == (5, 3)
    assert out["vacuity"].shape == (5,)
    assert (out["vacuity"] > 0).all() and (out["vacuity"] <= 1).all()


def test_evidence_computed_in_log_space_no_overflow(epc_head):
    torch.manual_seed(1)
    h = torch.randn(100, 32)
    out = epc_head(h)
    assert torch.isfinite(out["log_e"]).all()
    assert torch.isfinite(out["p_hat"]).all()
    assert (out["log_e"] <= 15.0 + 1e-3).all()
    assert (out["log_e"] >= -15.0 - 1e-3).all()


def test_register_class_zero_parameter_change(epc_head):
    """register_class must not alter any existing model parameter values."""
    epc_head.eval()
    h = torch.randn(20, 32)
    out_before = epc_head(h)
    state_before = {k: v.clone() for k, v in epc_head.state_dict().items()}

    new_samples = torch.randn(5, 32)
    epc_head.register_class("NewAttack", new_samples, n_sub=1)

    state_after = epc_head.state_dict()
    for key, before in state_before.items():
        after = state_after[key]
        # For the prototype bank the tensor grows; compare only original rows.
        if key.endswith("bank"):
            assert torch.equal(before, after[: before.shape[0]]), "Prototype bank changed during registration"
        else:
            assert torch.equal(before, after), f"Parameter {key} changed during registration"

    out_after = epc_head(h)
    # For the EPC head the logits are renormalised by the new class, so raw
    # logits change. The cosine similarities to old prototypes must be unchanged.
    assert torch.equal(out_before["cos_c"], out_after["cos_c"][:, :3])


def test_register_class_no_grad_enabled(epc_head):
    """The low-level prototype bank primitive must assert grad is disabled.

    EPCHead.register_class is a convenience wrapper that always enters
    torch.no_grad() itself; the contract is enforced at the PrototypeBank level.
    """
    epc_head.eval()
    new_samples = torch.randn(3, 32)
    with pytest.raises(AssertionError):
        with torch.enable_grad():
            epc_head.prototype_bank.register_class("NewAttack", new_samples)


def test_three_way_decision(epc_head):
    h = torch.zeros(4, 32)
    out = epc_head(h)
    decisions, margin = epc_head.decide(out, theta_unknown=0.5, theta_defer=0.1)
    assert decisions.shape == (4,)
    assert set(decisions.unique().tolist()).issubset({-2, -1, 0, 1, 2})


def test_tau_anneals(epc_head):
    epc_head.anneal_tau(0)
    tau0 = epc_head.tau.item()
    assert abs(tau0 - epc_head.tau_start) < 1e-3
    epc_head.anneal_tau(100)
    tau_final = epc_head.tau.item()
    assert abs(tau_final - epc_head.tau_final) < 1e-3
