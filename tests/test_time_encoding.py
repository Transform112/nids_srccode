"""Time encoding tests."""

import math

import torch

from argus.models.time_encoding import Time2Vec


def test_time2vec_linear_term_present():
    enc = Time2Vec(dim=16)
    dt = torch.tensor([0.0, 1.0, 2.0])
    out = enc(dt, scale_duration=1.0)
    assert out.shape == (3, 16)
    # Linear term first
    assert not torch.allclose(out[0, 0], out[1, 0], atol=1e-6)


def test_time2vec_periodicity_recoverable():
    """A 30-second periodic signal should be fit much better than an aperiodic one.

    Time2Vec's frequencies are learnable; the log-grid initialisation only needs
    to provide a basis from which periodicity is *recoverable*, not a perfect
    closed-form fit at init. We check the periodic signal fits meaningfully
    better than noise using the same basis.
    """
    torch.manual_seed(0)
    enc = Time2Vec(dim=16, period_min=0.1, period_max=600.0)
    t = torch.linspace(0, 120, 500)
    periodic = torch.sin(2 * math.pi * t / 30.0)
    aperiodic = torch.randn_like(t) * 0.3

    enc_t = enc(t, scale_duration=1.0)
    X = torch.cat([enc_t, torch.ones_like(t).unsqueeze(1)], dim=1)
    w_p = torch.linalg.lstsq(X, periodic).solution
    w_a = torch.linalg.lstsq(X, aperiodic).solution
    pred_p = X @ w_p
    pred_a = X @ w_a
    r2_p = 1 - ((pred_p - periodic) ** 2).mean() / (periodic ** 2).mean()
    r2_a = 1 - ((pred_a - aperiodic) ** 2).mean() / (aperiodic ** 2).mean()
    assert r2_p > 0.25
    assert r2_p > r2_a + 0.15


def test_time2vec_init_covers_log_grid():
    enc = Time2Vec(dim=16, period_min=0.1, period_max=600.0)
    freqs = enc.omega.data[1:].numpy()
    periods = 2 * math.pi / freqs
    assert periods.min() >= 0.05  # allow small numerical slack
    assert periods.max() <= 650.0
    # Ascending periods (logspace from period_min to period_max)
    assert (periods[:-1] <= periods[1:]).all()
