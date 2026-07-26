"""Configuration loading, merging, and validation.

Layering: default.yaml -> dataset/*.yaml -> model/*.yaml -> experiment/*.yaml -> CLI overrides.
Validator raises on unknown keys and enforces structural constraints from the spec.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from argus.constants import DATASET_STATS, MIN_UNIQUE_SRC_IP


SCHEMA_KEYS: set[str] = {
    "run",
    "paths",
    "data",
    "features",
    "graph",
    "model",
    "head",
    "regularisation",
    "loss",
    "train",
    "gates",
    "eval",
    "attack",
    "xai",
    "classes",
}

# Nested allowed keys for sections that are not dataset-specific.
SECTION_SCHEMA: dict[str, set[str]] = {
    "run": {
        "seed", "device", "precision", "deterministic", "out_dir", "registry",
    },
    "paths": {
        "dataset_dir", "interim_dir", "processed_dir", "artifact_dir", "cache_dir",
    },
    "data": {
        "dataset", "protocol", "subsample_target", "minority_threshold",
        "benign_floor_fraction", "dedup", "drop_null_labels", "time_bins_for_subsample",
        "holdout_size", "holdout_repeats", "holdout_seed", "node_granularity",
    },
    "features": {
        "te1_enabled", "te2_enabled", "te7_enabled", "quantile_n", "quantile_subsample",
        "clip_post_transform", "eps", "protocol_topk", "l7_proto_topk", "dst_port_topk",
        "spectral_nbins", "spectral_min_flows",
    },
    "graph": {
        "node_granularity", "anchor_bin_seconds", "window_short_seconds",
        "window_mid_seconds", "window_long_seconds", "neighbour_cap", "sampling",
        "strata", "memory_evict_seconds",
    },
    "model": {
        "name", "d_h", "d_A", "d_B", "d_t", "d_z", "layers", "heads", "aggregation",
        "trim_beta", "soft_medoid_temp", "multi_aggregator", "time_encoding",
        "time2vec_period_min", "time2vec_period_max", "norm_node", "norm_mlp",
        "prenorm", "te3_enabled", "te4_enabled", "te5_enabled", "te6_enabled",
    },
    "head": {
        "type", "sub_prototypes_benign", "sub_prototypes_attack_large",
        "sub_prototypes_attack_small", "margin_m", "tau_start", "tau_final",
        "tau_anneal_epochs", "tau_min", "log_evidence_clamp", "am_softmax_margin",
        "am_softmax_scale_start", "am_softmax_scale_final", "am_softmax_warmup_epochs",
        "fp32_head", "theta_unknown", "theta_defer", "target_false_unknown_rate",
        "target_defer_rate",
    },
    "regularisation": {
        "dropout", "droppath", "dropedge", "edge_feature_dropout_a",
        "edge_feature_dropout_b", "memory_dropout", "weight_decay", "label_smoothing",
    },
    "loss": {
        "lambda_compact", "lambda_channel", "channel_ratio_tolerance",
        "channel_penalty_stride", "lambda_div", "lambda_unknown", "lambda_kl_max",
        "kl_anneal_epochs", "synth_unknown_ratio", "mixup_mu_low", "mixup_mu_high",
        "cos_reject", "structural_candidates", "effective_number_nu",
    },
    "train": {
        "stage1_epochs", "stage1_lr", "stage1_patience", "stage2_epochs", "stage2_lr",
        "stage2_patience", "stage3_joint_finetune", "batch_anchor_bins", "n_per_class",
        "min_classes_per_batch", "bptt_chunk", "warmup_epochs", "grad_clip",
        "min_count_for_prototype", "checkpoint_every_epoch",
    },
    "gates": {
        "g0_capacity_check", "g0_subset_size", "g0_required_train_acc",
        "g1_min_val_f1_epoch5", "g2_max_proto_cosine", "g3_max_known_vacuity",
        "g4_min_unknown_vacuity", "g5_max_channel_ratio", "g7_max_train_val_gap",
        "g8_max_collapsed_classes", "monitor_every_steps",
    },
    "eval": {
        "metrics_seeds", "bootstrap_resamples", "ece_bins", "ece_binning",
        "few_shot_n", "openness_holdout_sizes",
    },
    "attack": {
        "a1_steps", "a1_epsilons", "a2_budgets", "a2_spread", "a3_poison_rates",
        "a4_steps", "a5_jitter_sigmas",
    },
    "xai": {
        "topk_edges", "topk_features", "ig_steps", "gnnexplainer_epochs",
        "pgexplainer_epochs", "shap_nsamples", "stability_n_perturb",
    },
    "classes": {
        "benign", "attacks", "canonical", "families",
    },
}


def load_config(
    base_dir: Path | str = Path(__file__).parents[2] / "config",
    dataset: str | None = None,
    model: str | None = None,
    experiment: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and merge configuration layers.

    Args:
        base_dir: directory containing default.yaml and subfolders.
        dataset: name of config/dataset/<name>.yaml.
        model: name of config/model/<name>.yaml.
        experiment: name of config/experiment/<name>.yaml.
        overrides: CLI-style overrides as "key=value" strings.
    """
    base_dir = Path(base_dir)
    cfgs = [OmegaConf.load(base_dir / "default.yaml")]
    if dataset:
        cfgs.append(OmegaConf.load(base_dir / "dataset" / f"{dataset}.yaml"))
    if model:
        cfgs.append(OmegaConf.load(base_dir / "model" / f"{model}.yaml"))
    if experiment:
        cfgs.append(OmegaConf.load(base_dir / "experiment" / f"{experiment}.yaml"))
    if overrides:
        cfgs.append(OmegaConf.from_dotlist(overrides))
    cfg: DictConfig = OmegaConf.merge(*cfgs)
    OmegaConf.set_struct(cfg, True)
    validate(cfg)
    return cfg


def validate(cfg: DictConfig) -> None:
    """Validate merged config; raise ValueError on any violation."""
    # Unknown keys (top-level and nested)
    for key in cfg.keys():
        if key not in SCHEMA_KEYS:
            raise ValueError(f"Unknown top-level config key: {key}")
    for section, allowed in SECTION_SCHEMA.items():
        if section not in cfg:
            continue
        for key in cfg[section].keys():
            if key not in allowed:
                raise ValueError(f"Unknown key in config.{section}: {key}")

    # Channel widths
    if cfg.model.d_A + cfg.model.d_B != cfg.model.d_h:
        raise ValueError(
            f"model.d_A ({cfg.model.d_A}) + model.d_B ({cfg.model.d_B}) "
            f"must equal model.d_h ({cfg.model.d_h})"
        )

    # Head divisibility
    if cfg.model.d_h % cfg.model.heads != 0:
        raise ValueError(
            f"model.d_h ({cfg.model.d_h}) must be divisible by model.heads ({cfg.model.heads})"
        )

    # Window ordering
    if not (
        cfg.graph.window_short_seconds
        < cfg.graph.window_mid_seconds
        < cfg.graph.window_long_seconds
    ):
        raise ValueError("Window sizes must satisfy short < mid < long")

    # Anchor bin
    if cfg.graph.anchor_bin_seconds > cfg.graph.window_short_seconds:
        raise ValueError("anchor_bin_seconds must be <= window_short_seconds")

    # Trim validity
    if not 0 <= cfg.model.trim_beta < 0.5:
        raise ValueError("trim_beta must be in [0, 0.5)")

    # Temperatures
    if cfg.head.tau_final < cfg.head.tau_min:
        raise ValueError("tau_final must be >= tau_min")
    if cfg.head.tau_start < cfg.head.tau_final:
        raise ValueError("tau_start must be >= tau_final")

    # BatchNorm ban
    norm_node = str(cfg.model.norm_node).lower()
    norm_mlp = str(cfg.model.norm_mlp).lower()
    if "batchnorm" in norm_node or "batch_norm" in norm_node:
        raise ValueError("model.norm_node must not be batchnorm")
    if "batchnorm" in norm_mlp or "batch_norm" in norm_mlp:
        raise ValueError("model.norm_mlp must not be batchnorm")

    # TRAP-1 guard
    if cfg.graph.node_granularity == "ip":
        ds = cfg.data.dataset
        stats = DATASET_STATS.get(ds)
        if stats and stats["src_ips"] < MIN_UNIQUE_SRC_IP:
            raise ValueError(
                f"TRAP-1: dataset {ds} has only {stats['src_ips']} unique source IPs "
                f"(< {MIN_UNIQUE_SRC_IP}). Use node_granularity='ip_port'."
            )

    # Sub-prototypes
    if cfg.head.sub_prototypes_benign < 1:
        raise ValueError("sub_prototypes_benign must be >= 1")
    if cfg.head.sub_prototypes_attack_large < 1:
        raise ValueError("sub_prototypes_attack_large must be >= 1")
    if cfg.head.sub_prototypes_attack_small < 1:
        raise ValueError("sub_prototypes_attack_small must be >= 1")

    # Precision
    if str(cfg.run.precision).lower() == "fp16" and not cfg.head.fp32_head:
        raise ValueError("head.fp32_head must be true when precision=fp16")

    # Protocol B2 requires families
    if cfg.data.protocol == "B2" and "families" not in (cfg.get("classes") or {}):
        raise ValueError("Protocol B2 requires class families to be defined in dataset config")


def resolved_path(cfg: DictConfig, key: str) -> Path:
    """Resolve a path key relative to the project root."""
    base = Path(__file__).parents[2]
    return (base / cfg.paths[key]).resolve()


@dataclass(frozen=True)
class ScaleConfig:
    """Helper to access per-scale durations."""

    short: int
    mid: int
    long: int

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "ScaleConfig":
        return cls(
            short=cfg.graph.window_short_seconds,
            mid=cfg.graph.window_mid_seconds,
            long=cfg.graph.window_long_seconds,
        )

    def duration(self, name: str) -> int:
        return getattr(self, name)

    def all(self) -> list[tuple[str, int]]:
        return [("short", self.short), ("mid", self.mid), ("long", self.long)]
