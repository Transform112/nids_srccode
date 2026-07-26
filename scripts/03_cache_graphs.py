"""03 — Precompute graph batch cache (single-process, compressed).

Builds anchor-bin batches for train/val/test splits and serialises them
as compressed ``.pt.gz`` files.  Training scripts then replay from cache at
~50 ms/bin instead of rebuilding at ~4 sec/bin.

Single-process avoids PyTorch multiprocessing race conditions.  gzip
compression cuts disk usage ~3× (6 GB → 2 GB for a typical split).

Usage:
    python scripts/03_cache_graphs.py --dataset cicids2018
    python scripts/03_cache_graphs.py --dataset cicids2018 --splits train val
    python scripts/03_cache_graphs.py --dataset cicids2018 \\
        --set graph.anchor_bin_seconds=10 --set graph.window_short_seconds=10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from argus.config import load_config, resolved_path
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.graph.cache import _save_compressed, verify_or_write_cache_meta
from argus.graph.node_features import node_feature_dim
from argus.utils.io import derive_class_vocab, holdout_subdir, run_suffix


def build_source(
    processed_dir: Path, split: str, cfg, feature_names: list[str],
    label_to_id: dict[str, int],
) -> AnchorBinGraphSource:
    df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    df["_label_id"] = df["canonical_label"].map(label_to_id)

    src_ids, dst_ids, _ = assign_node_ids(
        df, node_granularity=cfg.graph.node_granularity,
        src_ip_col="IPV4_SRC_ADDR", dst_ip_col="IPV4_DST_ADDR",
        src_port_col="L4_SRC_PORT", dst_port_col="L4_DST_PORT",
    )
    return AnchorBinGraphSource(
        df["FLOW_START_MILLISECONDS"].to_numpy(),
        df[feature_names].to_numpy(dtype=np.float32),
        src_ids, dst_ids, df["_label_id"].to_numpy(),
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


def run(
    dataset: str,
    splits: list[str] | None = None,
    overrides: list[str] | None = None,
    holdout_index: int | None = None,
) -> dict:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = holdout_subdir(resolved_path(cfg, "processed_dir") / dataset, holdout_index)
    artifact_dir = holdout_subdir(resolved_path(cfg, "artifact_dir") / dataset, holdout_index)

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    # Derive class vocab directly from data — do NOT read a persisted
    # class_vocab.json here. This script is meant to run *before*
    # 04_train_encoder.py (CPU cache-build session ahead of GPU training),
    # so that file won't exist yet on a fresh Kaggle session. Deriving it
    # the same way 04 does guarantees identical label ids either way.
    train_labels_df = pd.read_parquet(
        processed_dir / "train_features.parquet", columns=["canonical_label"]
    )
    class_names = derive_class_vocab(train_labels_df)
    label_to_id = {c: i for i, c in enumerate(class_names)}
    f_v = node_feature_dim(cfg.features.te7_enabled)

    stage1_dir = (Path(__file__).parents[1] / cfg.run.out_dir
                  / f"{dataset}_stage1{run_suffix(holdout_index)}")
    if splits is None:
        splits = ["train", "val"]

    cache_root = stage1_dir / "cache"
    report: dict[str, dict] = {}

    for split in splits:
        t_start = time.perf_counter()
        print(f"[03] Building source for {split} ...")
        source = build_source(processed_dir, split, cfg, feature_names, label_to_id)
        split_cache = cache_root / split
        verify_or_write_cache_meta(split_cache, cfg)
        bins = source.unique_bins
        n_bins = len(bins)
        print(f"[03] {split}: {n_bins} bins  |  anchor={cfg.graph.anchor_bin_seconds}s  "
              f"windows=({cfg.graph.window_short_seconds}/{cfg.graph.window_mid_seconds}/"
              f"{cfg.graph.window_long_seconds})s  |  K={cfg.graph.neighbour_cap}")

        built = 0
        skipped = 0
        errors = 0
        build_time = 0.0
        last_log = t_start

        for i, bin_id in enumerate(bins):
            cache_path = split_cache / f"bin_{bin_id:06d}.pt.gz"
            if cache_path.exists():
                skipped += 1
                # Remove stale uncompressed .pt files from previous broken runs
                old_path = split_cache / f"bin_{bin_id:06d}.pt"
                if old_path.exists():
                    old_path.unlink()
            else:
                t_bin = time.perf_counter()
                try:
                    batch = source.build_bin_batch(bin_id, f_v=f_v)
                    if batch is not None:
                        _save_compressed(batch, cache_path)
                        built += 1
                    else:
                        skipped += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"[03]   ERROR bin {bin_id}: {e}", flush=True)
                build_time += time.perf_counter() - t_bin

            # Progress every 500 bins or 30 seconds
            elapsed = time.perf_counter() - last_log
            if (i + 1) % 500 == 0 or elapsed > 30:
                pct = (i + 1) / n_bins * 100
                rate = (i + 1 - skipped) / max(time.perf_counter() - t_start, 1)
                eta_min = (n_bins - i - 1) / max(rate, 0.001) / 60
                print(f"[03]   {split} {i + 1}/{n_bins} bins ({pct:.0f}%)  "
                      f"built={built}  skip={skipped}  err={errors}  "
                      f"ETA={eta_min:.0f}min", flush=True)
                last_log = time.perf_counter()

        elapsed_m = (time.perf_counter() - t_start) / 60
        disk_mb = sum(f.stat().st_size for f in split_cache.glob("*.pt.gz")) / (1024**2)
        report[split] = {"bins": n_bins, "built": built, "skipped": skipped,
                         "errors": errors, "elapsed_min": round(elapsed_m, 1),
                         "disk_mb": round(disk_mb, 1)}
        print(f"[03] {split} done: {built} built, {errors} errors, "
              f"{elapsed_m:.0f}min, {disk_mb:.0f}MB on disk")

    print(f"[03] Cache saved to {cache_root}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Protocol B: cache holdout_b<i>'s splits into <dataset>_stage1_b<i>/cache")
    args = parser.parse_args()
    run(args.dataset, splits=args.splits, overrides=args.overrides,
        holdout_index=args.holdout_index)
