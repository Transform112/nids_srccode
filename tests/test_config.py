"""Config loading and validation tests."""

import pytest
from omegaconf import OmegaConf

from argus.config import load_config, validate


def test_load_default_config():
    cfg = load_config()
    assert cfg.model.d_h == 128
    assert cfg.model.d_A + cfg.model.d_B == cfg.model.d_h


def test_dataset_override():
    cfg = load_config(dataset="cicids2018")
    assert cfg.data.dataset == "cicids2018"


def test_trap1_guard_raises():
    with pytest.raises(ValueError, match="TRAP-1"):
        load_config(dataset="unsw_nb15", overrides=["graph.node_granularity=ip"])


def test_batchnorm_rejected():
    with pytest.raises(ValueError, match="batchnorm"):
        load_config(overrides=["model.norm_node=batchnorm"])


def test_channel_width_mismatch_rejected():
    with pytest.raises(ValueError, match="d_A"):
        load_config(overrides=["model.d_A=40", "model.d_B=80"])


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides=["model.unknown_key=1"])
