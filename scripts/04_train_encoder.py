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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import torch

from argus.config import load_config, resolved_path  # noqa: E402
from argus.constants import DATASET_STATS, MIN_UNIQUE_SRC_IP  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids, enforce_trap1_guard  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.checkpoint import save_checkpoint  # noqa: E402
from argus.train.stage1_encoder import train_stage1  # noqa: E402


def _load_source(processed_dir: Path, split: str, cfg, feature_names: list[str]) -> AnchorBinGraphSource:
    import pandas as pd

    df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)

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
    )


def run(dataset: str, overrides: list[str] | None = None, max_bins: int | None = None) -> Path:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

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
    class_names = sorted(train_df["canonical_label"].unique().tolist())
    label_to_id = {c: i for i, c in enumerate(class_names)}
    with open(artifact_dir / "class_vocab.json", "w") as f:
        json.dump(class_names, f, indent=2)

    for split in ("train", "val"):
        df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
        df["_label_id"] = df["canonical_label"].map(label_to_id)
        df.to_parquet(processed_dir / f"{split}_features.parquet", index=False)

    train_source = _load_source(processed_dir, "train", cfg, feature_names)
    val_source = _load_source(processed_dir, "val", cfg, feature_names)

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    class_counts = train_df["canonical_label"].value_counts().to_dict()
    model = ArgusModel(
        cfg, f_e=f_e, f_v=18, class_names=class_names, class_counts=class_counts
    ).to(device)

    print(f"[04] Training Stage 1 on {device} ({len(class_names)} classes, F_e={f_e}) ...")
    result = train_stage1(model, train_source, val_source, cfg, device, max_bins=max_bins)
    print(f"[04] Stage 1 best val accuracy proxy: {result['best_val_acc']:.4f}")

    run_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage1"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "stage1_final.pt"
    save_checkpoint(ckpt_path, model, torch.optim.AdamW(model.parameters()), epoch=len(result["history"]))
    with open(run_dir / "stage1_history.json", "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"[04] Saved checkpoint to {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--max-bins", type=int, default=None, help="Cap anchor bins per epoch (dev/smoke runs)")
    args = parser.parse_args()
    run(args.dataset, overrides=args.overrides, max_bins=args.max_bins)
