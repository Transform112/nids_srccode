"""11 — Run the adversarial evaluation (A1-A5) against a trained ARGUS model.

Requires scripts 01-05 to have already produced: cleaned/split parquet files,
a fitted feature pipeline, and a Stage-2 checkpoint (falls back to the Stage-1
checkpoint for A2/A3, which don't need the trained head to be meaningful, but
A1/A4/A5 report against whichever checkpoint is available).

Samples up to `--per-class` malicious flows per attack class from the test
split as A1/A4/A5 targets, and one flow per class for the A2 budget sweep.
A3 (prototype poisoning) does not need a specific target flow — it attacks
the trained prototype bank directly.

Usage (laptop dev run, bounded):
    python scripts/11_run_adversarial.py --dataset cicids2018 --per-class 2 --max-bins 200

Usage (Kaggle-scale full sweep):
    python scripts/11_run_adversarial.py --dataset cicids2018 --per-class 20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from argus.attacks.a1_feature_pgd import run_a1_epsilon_sweep  # noqa: E402
from argus.attacks.a2_structural_injection import run_a2_budget_sweep  # noqa: E402
from argus.attacks.a3_prototype_poison import run_a3_poison_sweep  # noqa: E402
from argus.attacks.a4_adaptive import run_a4_adaptive  # noqa: E402
from argus.attacks.a5_temporal_jitter import run_a5_jitter_sweep  # noqa: E402
from argus.config import load_config, resolved_path  # noqa: E402
from argus.features.pipeline import FeaturePipeline  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.checkpoint import load_checkpoint  # noqa: E402


def _build_source(df: pd.DataFrame, cfg, feature_names: list[str]) -> AnchorBinGraphSource:
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


def _bin_and_target_pos(source: AnchorBinGraphSource, row_pos: int) -> tuple[int, int] | None:
    """Given a row position in the (already time-sorted) source arrays, return
    (bin_id, target_index_within_bin) if that row is a target of its bin."""
    bin_id = int(source.bin_ids[row_pos])
    ranges = dict((b, (lo, hi)) for b, lo, hi in source.ranges["short"])
    if bin_id not in ranges:
        return None
    lo, hi = ranges[bin_id]
    target_positions = np.nonzero(source.bin_ids[lo:hi] == bin_id)[0] + lo
    matches = np.nonzero(target_positions == row_pos)[0]
    if len(matches) == 0:
        return None
    return bin_id, int(matches[0])


def run(
    dataset: str,
    per_class: int = 5,
    max_bins: int | None = None,
    overrides: list[str] | None = None,
    seed: int = 0,
) -> dict:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]

    with open(artifact_dir / "class_vocab.json") as f:
        class_names = json.load(f)
    label_to_id = {c: i for i, c in enumerate(class_names)}
    benign_class_id = label_to_id[cfg.classes.benign]

    pipeline = FeaturePipeline.load(artifact_dir / "feature_pipeline.joblib")

    raw_test = pd.read_parquet(processed_dir / "test.parquet")
    raw_test = raw_test.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)

    feat_test = pd.read_parquet(processed_dir / "test_features.parquet")
    feat_test = feat_test.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    feat_test["_label_id"] = feat_test["canonical_label"].map(label_to_id)

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    model = ArgusModel(cfg, f_e=f_e, f_v=18, class_names=class_names).to(device)

    ckpt_dir = Path(__file__).parents[1] / cfg.run.out_dir
    stage2_ckpt = ckpt_dir / f"{dataset}_stage2" / "stage2_final.pt"
    stage1_ckpt = ckpt_dir / f"{dataset}_stage1" / "stage1_final.pt"
    if stage2_ckpt.exists():
        load_checkpoint(stage2_ckpt, model)
        print(f"[11] Loaded Stage-2 checkpoint from {stage2_ckpt}")
    elif stage1_ckpt.exists():
        load_checkpoint(stage1_ckpt, model)
        print(f"[11] Stage-2 checkpoint not found; loaded Stage-1 from {stage1_ckpt}")
    else:
        raise FileNotFoundError("No trained checkpoint found; run scripts/04 and 05 first.")

    source = _build_source(feat_test, cfg, feature_names)
    benign_pool = feat_test.loc[
        feat_test["canonical_label"] == cfg.classes.benign, feature_names
    ].to_numpy(dtype=np.float32)
    if len(benign_pool) == 0:
        raise ValueError("No benign flows found in the test split for A2's injection pool.")

    rng = np.random.default_rng(seed)
    results: dict[str, list] = {"a1": [], "a2": [], "a4": [], "a5": [], "a3": []}

    attack_classes = [c for c in class_names if c != cfg.classes.benign]

    for attack_class in attack_classes:
        idxs = np.nonzero(feat_test["canonical_label"].to_numpy() == attack_class)[0]
        if len(idxs) == 0:
            continue
        sample_idxs = rng.choice(idxs, size=min(per_class, len(idxs)), replace=False)

        for row_pos in sample_idxs:
            found = _bin_and_target_pos(source, int(row_pos))
            if found is None:
                continue
            bin_id, target_idx = found
            batch = source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            raw_row = raw_test.iloc[[int(row_pos)]]

            a1_results = run_a1_epsilon_sweep(
                model, pipeline, batch, raw_row, feature_names, target_idx, device,
                epsilons=list(cfg.attack.a1_epsilons), steps=cfg.attack.a1_steps,
                benign_class_id=benign_class_id, seed=seed,
            )
            results["a1"].extend(
                {"attack_class": attack_class, "row": int(row_pos), **asdict(r)} for r in a1_results
            )

            a5_results = run_a5_jitter_sweep(
                model, pipeline, batch, raw_row, feature_names, target_idx, device,
                sigmas=list(cfg.attack.a5_jitter_sigmas), benign_class_id=benign_class_id, seed=seed,
            )
            results["a5"].extend(
                {"attack_class": attack_class, "row": int(row_pos), **asdict(r)} for r in a5_results
            )

            for objective in ("evasion", "unknown_avoidance"):
                a4_result = run_a4_adaptive(
                    model, pipeline, batch, raw_row, feature_names, target_idx, device,
                    objective=objective, steps=cfg.attack.a4_steps,
                    benign_class_id=benign_class_id, seed=seed,
                )
                results["a4"].append({"attack_class": attack_class, "row": int(row_pos), **asdict(a4_result)})

        # One A2 structural-injection sweep per class (headline attack).
        first_idx = int(sample_idxs[0]) if len(sample_idxs) else int(idxs[0])
        found = _bin_and_target_pos(source, first_idx)
        if found is not None:
            bin_id, target_idx = found
            a2_results = run_a2_budget_sweep(
                model, source, bin_id, target_idx,
                victim_node_id=int(source.dst_ids[first_idx]), injection_host_id=10**9,
                benign_pool=benign_pool, device=device,
                budgets=list(cfg.attack.a2_budgets), spread=cfg.attack.a2_spread,
                benign_class_id=benign_class_id,
            )
            results["a2"].extend(
                {"attack_class": attack_class, "row": first_idx, **asdict(r)} for r in a2_results
            )

        # A3: attacks the trained prototype bank directly, independent of any target flow.
        if hasattr(model.head, "prototype_bank"):
            class_idx = label_to_id[attack_class]
            for gate_enabled in (True, False):
                for poison_rate in cfg.attack.a3_poison_rates:
                    a3_result = run_a3_poison_sweep(
                        model.head, class_idx, poison_rate=poison_rate, momentum=0.99,
                        n_steps=200, theta_unknown=cfg.head.theta_unknown or 0.5,
                        gate_enabled=gate_enabled, seed=seed,
                    )
                    results["a3"].append({"attack_class": attack_class, **asdict(a3_result)})

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for name in ("a1", "a2", "a4", "a5"):
        n = len(results[name])
        evaded = sum(1 for r in results[name] if r.get("evaded"))
        print(f"[11] {name}: {n} evaluations, {evaded} evaded ({evaded / n:.1%})" if n else f"[11] {name}: 0 evaluations")
    print(f"[11] a3: {len(results['a3'])} poison-sweep configs")
    print(f"[11] Wrote {out_dir / 'adversarial_results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--max-bins", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, per_class=args.per_class, max_bins=args.max_bins, overrides=args.overrides, seed=args.seed)
