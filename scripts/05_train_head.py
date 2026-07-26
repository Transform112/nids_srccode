"""05 — Stage-2 evidential head training. Loads the Stage-1 checkpoint.

Usage:
    python scripts/05_train_head.py --dataset cicids2018 --set run.device=cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import torch

from argus.config import load_config, resolved_path  # noqa: E402
from argus.graph.node_features import node_feature_dim  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from argus.train.stage2_head import train_stage2  # noqa: E402
from argus.train.thresholds import calibrate_thresholds  # noqa: E402
from argus.utils.io import holdout_subdir, resolve_class_vocab, run_suffix  # noqa: E402
from argus.utils.registry import RunRegistry  # noqa: E402

# Reuse the Stage-1 source-building helper without making `scripts` a package.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_encoder_04", Path(__file__).parent / "04_train_encoder.py"
)
_train_encoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_encoder)
_load_source = _train_encoder._load_source


def run(dataset: str, overrides: list[str] | None = None, max_bins: int | None = None,
        resume: bool = False, force: bool = False, holdout_index: int | None = None) -> Path:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = holdout_subdir(resolved_path(cfg, "processed_dir") / dataset, holdout_index)
    artifact_dir = holdout_subdir(resolved_path(cfg, "artifact_dir") / dataset, holdout_index)
    suffix = run_suffix(holdout_index)

    run_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage2{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{dataset}_stage2{suffix}_seed{cfg.run.seed}"
    registry = RunRegistry(Path(__file__).parents[1] / cfg.run.out_dir / "registry.jsonl")
    ckpt_path = run_dir / "stage2_final.pt"
    if registry.is_complete(run_id) and ckpt_path.exists() and not force:
        print(f"[05] Run '{run_id}' already registered as complete and {ckpt_path} exists "
              f"— skipping (use --force to retrain).")
        return ckpt_path

    manifest_path = artifact_dir / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]

    vocab_path = resolve_class_vocab(cfg, dataset, artifact_dir, holdout_index=holdout_index)
    with open(vocab_path) as f:
        class_names = json.load(f)

    label_to_id = {c: i for i, c in enumerate(class_names)}

    import pandas as pd
    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    class_counts = train_df["canonical_label"].value_counts().to_dict()

    # Reuse Stage-1 graph cache if it exists; otherwise build fresh.
    cache_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage1{suffix}" / "cache"
    if not cache_dir.is_dir():
        cache_dir = None  # fall back to building from scratch
    else:
        print(f"[05] Reusing Stage-1 graph cache: {cache_dir}")
    train_source = _load_source(processed_dir, "train", cfg, feature_names, label_to_id, cache_dir)
    val_source = _load_source(processed_dir, "val", cfg, feature_names, label_to_id, cache_dir)

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    f_v = node_feature_dim(cfg.features.te7_enabled)
    model = ArgusModel(
        cfg, f_e=f_e, f_v=f_v, class_names=class_names, class_counts=class_counts
    ).to(device)

    stage1_ckpt = (Path(__file__).parents[1] / cfg.run.out_dir
                   / f"{dataset}_stage1{suffix}" / "stage1_final.pt")
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Run 04_train_encoder.py first: {stage1_ckpt} not found")
    load_checkpoint(stage1_ckpt, model)
    print(f"[05] Loaded Stage-1 checkpoint from {stage1_ckpt}")

    print(f"[05] Training Stage 2 on {device} ...")
    result = train_stage2(model, train_source, val_source, cfg, device, max_bins=max_bins,
                          run_dir=run_dir, resume=resume)
    print(f"[05] Stage 2 best val accuracy proxy: {result['best_val_acc']:.4f}")

    thresholds = calibrate_thresholds(
        model, val_source, device,
        target_false_unknown_rate=cfg.head.target_false_unknown_rate,
        target_defer_rate=cfg.head.target_defer_rate,
        max_bins=max_bins,
    )
    print(f"[05] Calibrated thresholds: {thresholds}")

    save_checkpoint(
        ckpt_path, model, result["optimizer"], epoch=len(result["history"]),
        extra={"thresholds": thresholds, "best_epoch": result["best_epoch"],
               "best_val_acc": result["best_val_acc"]},
    )
    with open(run_dir / "stage2_history.json", "w") as f:
        json.dump(result["history"], f, indent=2)
    with open(run_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[05] Saved checkpoint to {ckpt_path}")

    registry.register(
        run_id, dataset=dataset, protocol=cfg.data.protocol, model="argus_stage2",
        metrics={"best_val_acc": result["best_val_acc"], "best_epoch": result["best_epoch"],
                 "epochs_run": len(result["history"]), "thresholds": thresholds},
    )
    print(f"[05] Registered run '{run_id}' in registry.jsonl")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--max-bins", type=int, default=None, help="Cap anchor bins per epoch (dev/smoke runs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from stage2_ckpt_last.pt in the run dir if present")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if this run is already registered as complete")
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Protocol B: train on holdout_b<i>'s splits into <dataset>_stage2_b<i>")
    args = parser.parse_args()
    run(args.dataset, overrides=args.overrides, max_bins=args.max_bins,
        resume=args.resume, force=args.force, holdout_index=args.holdout_index)
