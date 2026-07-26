"""04 — Stage-1 encoder training.

Usage (laptop CPU smoke test):
    python scripts/04_train_encoder.py --dataset cicids2018 --set run.device=cpu \
        --set train.stage1_epochs=2

Usage (Kaggle GPU):
    python scripts/04_train_encoder.py --dataset cicids2018 --set run.device=cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import torch

from argus.config import load_config, resolved_path  # noqa: E402
from argus.constants import DATASET_STATS, MIN_UNIQUE_SRC_IP  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids, enforce_trap1_guard  # noqa: E402
from argus.graph.cache import CachedGraphSource  # noqa: E402
from argus.graph.node_features import node_feature_dim  # noqa: E402
from argus.graph.windows import assign_anchor_bins  # noqa: E402
from argus.losses.class_balance import effective_target_counts  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.checkpoint import save_checkpoint  # noqa: E402
from argus.train.stage1_encoder import train_stage1  # noqa: E402
from argus.utils.io import derive_class_vocab, holdout_subdir, run_suffix  # noqa: E402
from argus.utils.registry import RunRegistry  # noqa: E402


def _source_from_df(df, cfg, feature_names: list[str],
                    label_to_id: dict[str, int] | None = None) -> AnchorBinGraphSource:
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)

    # Add _label_id in-memory — avoids writing back to parquet (Kaggle mounts are read-only).
    if label_to_id is not None:
        df = df.assign(_label_id=df["canonical_label"].map(label_to_id))

    src_ids, dst_ids, _ = assign_node_ids(
        df, node_granularity=cfg.graph.node_granularity,
        src_ip_col="IPV4_SRC_ADDR", dst_ip_col="IPV4_DST_ADDR",
        src_port_col="L4_SRC_PORT", dst_port_col="L4_DST_PORT",
    )
    times_ms = df["FLOW_START_MILLISECONDS"].to_numpy()
    edge_features = df[feature_names].to_numpy(dtype=np.float32)
    label_ids = df["_label_id"].to_numpy()

    return AnchorBinGraphSource(
        times_ms, edge_features, src_ids, dst_ids, label_ids,
        anchor_bin_seconds=cfg.graph.anchor_bin_seconds,
        window_short_seconds=cfg.graph.window_short_seconds,
        window_mid_seconds=cfg.graph.window_mid_seconds,
        window_long_seconds=cfg.graph.window_long_seconds,
        neighbour_cap=cfg.graph.neighbour_cap,
        sampling=cfg.graph.sampling,
        strata=cfg.graph.strata,
        te7_enabled=cfg.features.te7_enabled,
        spectral_nbins=cfg.features.spectral_nbins,
        spectral_min_flows=cfg.features.spectral_min_flows,
    )


def _load_source(processed_dir: Path, split: str, cfg, feature_names: list[str],
                 label_to_id: dict[str, int] | None = None,
                 cache_dir: Path | None = None) -> AnchorBinGraphSource | CachedGraphSource:
    import pandas as pd

    df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
    source = _source_from_df(df, cfg, feature_names, label_to_id)
    if cache_dir is not None:
        split_cache = cache_dir / split
        source = CachedGraphSource(source, cache_dir=split_cache, cfg=cfg, label=f"{split}")
    return source


def _gate0_preflight(dataset: str, overrides: list[str] | None, train_df, feature_names: list[str],
                     f_e: int, class_names: list[str], label_to_id: dict[str, int],
                     class_counts: dict, device: torch.device, run_dir: Path) -> None:
    """Gate G0: 1,000-sample overfit capacity check (docs/06_TRAINING.md §8.1).

    All regularisation disabled, throwaway model, trained until it memorises
    the subset or runs out of attempts. Hard-fails the run if capacity is
    missing — cheaper to find out in minutes than after a 12h GPU session.
    """
    from argus.losses.am_softmax import AMSoftmaxLoss, CompactnessLoss
    from argus.train.gates import gate_g0_capacity, record_gate
    from argus.train.loop import run_epoch
    from argus.train.stage1_encoder import make_stage1_loss_fn

    g0_cfg = load_config(dataset=dataset, model="argus", overrides=(overrides or []) + [
        "regularisation.dropout=0.0",
        "regularisation.droppath=0.0",
        "regularisation.weight_decay=0.0",
        "regularisation.label_smoothing=0.0",
        "regularisation.memory_dropout=0.0",
    ])
    subset_size = getattr(g0_cfg.gates, "g0_subset_size", 1000)
    required = getattr(g0_cfg.gates, "g0_required_train_acc", 0.99)

    quota = max(subset_size // max(len(class_names), 1), 1)
    subset = train_df.groupby("canonical_label", group_keys=False).head(quota)
    if len(subset) > subset_size:
        subset = subset.sort_values("FLOW_START_MILLISECONDS").head(subset_size)
    print(f"[04] G0 preflight: {len(subset)} flows across "
          f"{subset['canonical_label'].nunique()} classes, regularisation off ...", flush=True)

    source = _source_from_df(subset, g0_cfg, feature_names, label_to_id)
    torch.manual_seed(g0_cfg.run.seed)
    f_v = node_feature_dim(g0_cfg.features.te7_enabled)
    model = ArgusModel(
        g0_cfg, f_e=f_e, f_v=f_v, class_names=class_names, class_counts=class_counts
    ).to(device)
    base_lr = g0_cfg.train.stage1_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=0.0)
    am_loss = AMSoftmaxLoss(
        margin=g0_cfg.head.am_softmax_margin,
        scale_start=g0_cfg.head.am_softmax_scale_start,
        scale_final=g0_cfg.head.am_softmax_scale_final,
        warmup_epochs=g0_cfg.head.am_softmax_warmup_epochs,
        label_smoothing=0.0,
    )
    compact_loss = CompactnessLoss()

    acc = 0.0
    # A pure memorisation check: give it enough epochs and cosine-decay the LR
    # to 0.1x for tail stability. Empirically this subset crosses 0.99 around
    # epoch ~80 with the fixed features (docs/BUGS.md #45/#46); the old 60-epoch
    # constant-LR loop peaked at ~0.98 and oscillated, false-failing G0. This
    # schedule is local to the throwaway preflight and does not touch the
    # production Stage-1 optimiser.
    max_g0_epochs = 150
    for epoch in range(max_g0_epochs):
        cur_lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * epoch / max_g0_epochs)))
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        loss_fn = make_stage1_loss_fn(
            am_loss, compact_loss, g0_cfg.loss.lambda_compact, g0_cfg.loss.lambda_div, epoch)
        result = run_epoch(source, model, optimizer, loss_fn, device, train=True,
                           grad_clip=g0_cfg.train.grad_clip,
                           bptt_chunk=g0_cfg.train.bptt_chunk,
                           log_every_bins=10**9)
        acc = result.accuracy
        if acc >= required:
            break

    passed, msg = gate_g0_capacity(acc, required)
    record_gate(run_dir / "gates_report.json", "G0", passed,
                msg + f" (after {epoch + 1} epochs on {len(subset)} flows)")
    if not passed:
        raise RuntimeError(
            f"Gate G0 failed: model cannot memorise {len(subset)} flows "
            f"(train_acc={acc:.4f} < {required}). Do not spend GPU hours on this "
            f"configuration — see docs/06_TRAINING.md §8.1."
        )


def run(dataset: str, overrides: list[str] | None = None, max_bins: int | None = None,
        resume: bool = False, force: bool = False, skip_gate0: bool = False,
        holdout_index: int | None = None) -> Path:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = holdout_subdir(resolved_path(cfg, "processed_dir") / dataset, holdout_index)
    artifact_dir = holdout_subdir(resolved_path(cfg, "artifact_dir") / dataset, holdout_index)
    suffix = run_suffix(holdout_index)

    stats = DATASET_STATS.get(dataset, {})
    enforce_trap1_guard(
        cfg.graph.node_granularity, stats.get("src_ips", 10**9), MIN_UNIQUE_SRC_IP,
    )

    manifest_path = artifact_dir / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]

    import pandas as pd
    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    class_names = derive_class_vocab(train_df)
    label_to_id = {c: i for i, c in enumerate(class_names)}

    # Write class_vocab to run_dir (always writable, even on Kaggle read-only mounts).
    run_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage1{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{dataset}_stage1{suffix}_seed{cfg.run.seed}"
    registry = RunRegistry(Path(__file__).parents[1] / cfg.run.out_dir / "registry.jsonl")
    ckpt_path = run_dir / "stage1_final.pt"
    if registry.is_complete(run_id) and ckpt_path.exists() and not force:
        print(f"[04] Run '{run_id}' already registered as complete and {ckpt_path} exists "
              f"— skipping (use --force to retrain).")
        return ckpt_path
    cache_dir = run_dir / "cache"
    print(f"[04] Graph cache dir: {cache_dir}")
    with open(run_dir / "class_vocab.json", "w") as f:
        json.dump(class_names, f, indent=2)
    # Also try artifact_dir (local dev); ignore if read-only (Kaggle).
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(artifact_dir / "class_vocab.json", "w") as f:
            json.dump(class_names, f, indent=2)
    except OSError:
        pass  # Kaggle read-only mount — run_dir copy is canonical

    train_source = _load_source(processed_dir, "train", cfg, feature_names, label_to_id, cache_dir)
    val_source = _load_source(processed_dir, "val", cfg, feature_names, label_to_id, cache_dir)

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    class_counts = train_df["canonical_label"].value_counts().to_dict()

    # Gate G0 preflight (docs/06_TRAINING.md §8.1) — skipped on resume (the
    # architecture already passed before the interrupted run started).
    g0_enabled = bool(getattr(cfg.gates, "g0_capacity_check", True))
    if g0_enabled and not skip_gate0 and not resume:
        _gate0_preflight(dataset, overrides, train_df, feature_names, f_e,
                         class_names, label_to_id, class_counts, device, run_dir)
    else:
        print("[04] G0 preflight skipped "
              f"(enabled={g0_enabled}, skip_flag={skip_gate0}, resume={resume})")

    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    f_v = node_feature_dim(cfg.features.te7_enabled)
    model = ArgusModel(
        cfg, f_e=f_e, f_v=f_v, class_names=class_names, class_counts=class_counts
    ).to(device)

    print(f"[04] Training Stage 1 on {device} ({len(class_names)} classes, F_e={f_e}) ...")
    # Weights must describe the distribution the loss actually sees, which is
    # the post-cap one — capping moves ddos_hoic 27.3%->3.6% and infiltration
    # 7.6%->30.8% (docs/BUGS.md #51). Raw counts would weight those two
    # identically and point the correction the wrong way.
    n_per_class = getattr(cfg.train, "n_per_class", None)
    # assign_anchor_bins needs ascending times, and labels must stay aligned to
    # them — sort once and take both columns from the same ordering.
    by_time = train_df.sort_values("FLOW_START_MILLISECONDS")
    eff = effective_target_counts(
        by_time["canonical_label"].map(label_to_id).to_numpy(),
        assign_anchor_bins(by_time["FLOW_START_MILLISECONDS"].to_numpy(),
                           cfg.graph.anchor_bin_seconds),
        len(class_names), n_per_class,
    )
    weight_counts = {name: int(eff[i]) for i, name in enumerate(class_names)}
    print(f"[04] Loss targets per epoch after n_per_class={n_per_class} capping: "
          f"{sum(eff):,} of {len(train_df):,} flows", flush=True)

    result = train_stage1(model, train_source, val_source, cfg, device, max_bins=max_bins,
                          run_dir=run_dir, resume=resume, class_counts=class_counts,
                          weight_counts=weight_counts)
    print(f"[04] Stage 1 best val macro-F1: {result['best_val_macro_f1']:.4f} "
          f"(epoch {result['best_epoch']})")
    for name, f1 in sorted(result["best_val_per_class_f1"].items(), key=lambda kv: kv[1]):
        print(f"[04]   {name:22s} F1={f1:.4f}")
    if result["excluded_classes"]:
        print(f"[04] Minimum-count classes left to few-shot registration: "
              f"{result['excluded_classes']} (docs/06_TRAINING.md sec 4.3)")

    # Log cache stats
    for name, src in [("train", train_source), ("val", val_source)]:
        if hasattr(src, 'stats'):
            s = src.stats()
            print(f"[04] Graph cache [{name}]: {s['cache_hits']} hits, "
                  f"{s['cache_misses']} misses, "
                  f"build={s['build_time_s']}s, load={s['load_time_s']}s")

    # Final checkpoint carries the model's best-epoch weights (restored inside
    # train_stage1) and the REAL trained optimizer state, so downstream resumes
    # and analyses see genuine training state.
    save_checkpoint(ckpt_path, model, result["optimizer"], epoch=len(result["history"]),
                    extra={"best_epoch": result["best_epoch"],
                           "best_val_macro_f1": result["best_val_macro_f1"],
                           "excluded_classes": result["excluded_classes"]})
    with open(run_dir / "stage1_history.json", "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"[04] Saved checkpoint to {ckpt_path}")

    registry.register(
        run_id, dataset=dataset, protocol=cfg.data.protocol, model="argus_stage1",
        metrics={"best_val_macro_f1": result["best_val_macro_f1"],
                 "best_val_per_class_f1": result["best_val_per_class_f1"],
                 "best_epoch": result["best_epoch"],
                 "epochs_run": len(result["history"])},
    )
    print(f"[04] Registered run '{run_id}' in registry.jsonl")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--max-bins", type=int, default=None, help="Cap anchor bins per epoch (dev/smoke runs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from stage1_ckpt_last.pt in the run dir if present")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if this run is already registered as complete")
    parser.add_argument("--skip-gate0", action="store_true",
                        help="Skip the G0 overfit-capacity preflight (docs/06_TRAINING.md §8.1)")
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Protocol B: train on holdout_b<i>'s splits into <dataset>_stage1_b<i>")
    args = parser.parse_args()
    run(args.dataset, overrides=args.overrides, max_bins=args.max_bins,
        resume=args.resume, force=args.force, skip_gate0=args.skip_gate0,
        holdout_index=args.holdout_index)
