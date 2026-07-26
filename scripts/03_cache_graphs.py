"""03 — Precompute graph batch cache with multiprocessing.

Builds all anchor-bin batches for train/val/test splits in parallel across
all CPU cores and serialises them to disk as ``.pt`` files.  Training scripts
then replay from cache at ~50 ms/bin instead of rebuilding at ~5 sec/bin.

This is the difference between a 100-hour first epoch and a 30-second epoch.

Usage:
    python scripts/03_cache_graphs.py --dataset cicids2018
    python scripts/03_cache_graphs.py --dataset cicids2018 --splits train val
    python scripts/03_cache_graphs.py --dataset cicids2018 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from argus.config import load_config, resolved_path
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids

# Global refs for worker processes (avoids re-serialising the source per bin).
_worker_source: AnchorBinGraphSource | None = None
_worker_cache_dir: Path | None = None
_worker_f_v: int = 18


def _init_worker(source: AnchorBinGraphSource, cache_dir: Path, f_v: int) -> None:
    global _worker_source, _worker_cache_dir, _worker_f_v
    _worker_source = source
    _worker_cache_dir = cache_dir
    _worker_f_v = f_v


def _build_one(bin_id: int) -> tuple[int, bool, str]:
    """Build + save one bin. Returns (bin_id, success, error_msg)."""
    global _worker_source, _worker_cache_dir, _worker_f_v
    cache_path = _worker_cache_dir / f"bin_{bin_id:06d}.pt"
    if cache_path.exists():
        return (bin_id, True, "skip")
    try:
        batch = _worker_source.build_bin_batch(bin_id, _worker_f_v)
        if batch is not None:
            torch.save(batch, cache_path)
            return (bin_id, True, "ok")
        else:
            return (bin_id, True, "empty")
    except Exception as e:
        return (bin_id, False, str(e))


def build_source(
    processed_dir: Path, split: str, cfg, feature_names: list[str],
    label_to_id: dict[str, int] | None = None,
) -> AnchorBinGraphSource:
    """Build a fresh AnchorBinGraphSource for one split (worker-compatible)."""
    df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    if label_to_id is not None:
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
    )


def run(
    dataset: str,
    splits: list[str] | None = None,
    workers: int | None = None,
    overrides: list[str] | None = None,
) -> dict:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    # Try to load class_vocab; build label_to_id if available.
    label_to_id = None
    vocab_path = artifact_dir / "class_vocab.json"
    stage1_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage1"
    for p in [stage1_dir / "class_vocab.json", vocab_path]:
        if p.is_file():
            with open(p) as f:
                class_names = json.load(f)
            label_to_id = {c: i for i, c in enumerate(class_names)}
            break

    if splits is None:
        splits = ["train", "val", "test"]
    if workers is None:
        workers = max(1, cpu_count() - 2)

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage1" / "cache"
    report: dict[str, dict] = {}

    for split in splits:
        print(f"[03] Building source for {split} ...")
        source = build_source(processed_dir, split, cfg, feature_names, label_to_id)
        split_cache = out_dir / split
        split_cache.mkdir(parents=True, exist_ok=True)
        bins = source.unique_bins
        n_bins = len(bins)
        print(f"[03] {split}: {n_bins} bins, {workers} workers")

        # Multiprocessing precompute
        built = 0
        skipped = 0
        errors = 0
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(source, split_cache, 18)) as pool:
            for i, (bin_id, ok, status) in enumerate(
                pool.imap_unordered(_build_one, bins, chunksize=10)
            ):
                if ok and status == "skip":
                    skipped += 1
                elif ok:
                    built += 1
                else:
                    errors += 1
                    print(f"[03]   ERROR bin {bin_id}: {status}")
                if (i + 1) % 1000 == 0:
                    pct = (i + 1) / n_bins * 100
                    print(f"[03]   {split} {i + 1}/{n_bins} bins ({pct:.0f}%)  "
                          f"built={built}  skip={skipped}  err={errors}", flush=True)

        report[split] = {"bins": n_bins, "built": built, "skipped": skipped, "errors": errors}
        print(f"[03] {split} done: {built} built, {skipped} skipped, {errors} errors")

    print(f"[03] Cache saved to {out_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    run(args.dataset, splits=args.splits, workers=args.workers, overrides=args.overrides)
