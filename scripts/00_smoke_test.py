"""00 — Laptop smoke test: runs the full pipeline (01-05) on a small slice of the
real NF-CICIDS2018 CSV, with tiny epoch counts, entirely on CPU.

This is the fastest way to validate the whole pipeline works before scaling up
to Kaggle. Takes a few minutes on a laptop.

Usage:
    python scripts/00_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import importlib.util


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    dataset = "cicids2018"
    print("=" * 70)
    print("ARGUS laptop smoke test — dataset:", dataset)
    print("=" * 70)

    prepare_data = _load("prepare_data", "01_prepare_data.py")
    build_splits = _load("build_splits", "02_build_splits.py")
    fit_features = _load("fit_features", "03_fit_features.py")
    train_encoder = _load("train_encoder", "04_train_encoder.py")
    train_head = _load("train_head", "05_train_head.py")

    print("\n--- Step 1: prepare data (50,000 rows) ---")
    prepare_data.run(dataset, nrows=50_000)

    print("\n--- Step 2: build Protocol-A splits + audit ---")
    build_splits.run(dataset, protocol="A")

    print("\n--- Step 3: fit feature pipeline ---")
    fit_features.run(dataset)

    # run.out_dir is isolated from the real results/runs/<dataset>_stage1/
    # tree so a local smoke test never collides with a production Kaggle
    # cache/checkpoint directory (different graph.* settings would otherwise
    # trip the cache fingerprint guard in argus.graph.cache).
    smoke_overrides = [
        "run.device=cpu",
        "run.out_dir=results/runs_smoketest",
        "model.layers=1",
        "model.d_h=32",
        "model.d_A=12",
        "model.d_B=20",
        "model.d_z=16",
        "graph.neighbour_cap=8",
        "graph.window_mid_seconds=10",
        "graph.window_long_seconds=60",
    ]

    print("\n--- Step 4: Stage-1 encoder training (smoke config) ---")
    train_encoder.run(
        dataset,
        overrides=[*smoke_overrides, "train.stage1_epochs=1", "train.stage1_patience=1"],
        max_bins=30,
        skip_gate0=True,  # G0 preflight is a production check, not a smoke-test step
        force=True,       # smoke runs must never be skipped by an old registry entry
    )

    print("\n--- Step 5: Stage-2 evidential head training (smoke config) ---")
    train_head.run(
        dataset,
        overrides=[*smoke_overrides, "train.stage2_epochs=1", "train.stage2_patience=1"],
        max_bins=30,
        force=True,
    )

    print("\n" + "=" * 70)
    print("Smoke test complete. Pipeline runs end-to-end on a laptop.")
    print("=" * 70)


if __name__ == "__main__":
    main()
