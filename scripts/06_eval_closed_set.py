"""06 — Closed-set evaluation (Protocol A).

Computes the full closed-set metric suite (docs/08_EVALUATION.md §2) for a
trained ARGUS model on the Protocol-A test split. Produces per-class F1,
macro-F1, per-tier F1, and confusion matrix.

Usage:
    python scripts/06_eval_closed_set.py --dataset cicids2018
    python scripts/06_eval_closed_set.py --dataset cicids2018 --ckpt results/runs/my_run/stage2_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from argus.config import load_config, resolved_path
from argus.eval.metrics import closed_set_report, per_tier_macro_f1
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.models.argus import ArgusModel
from argus.train.loop import model_inputs_from_batch
from argus.train.checkpoint import load_checkpoint


def evaluate_closed_set(dataset: str, ckpt_path: str | Path | None = None) -> dict:
    cfg = load_config(dataset=dataset, model="argus")
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    with open(artifact_dir / "class_vocab.json") as f:
        class_names = json.load(f)

    # Build test source
    df = pd.read_parquet(processed_dir / "test_features.parquet")
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    src_ids, dst_ids, _ = assign_node_ids(
        df, node_granularity=cfg.graph.node_granularity,
        src_ip_col="IPV4_SRC_ADDR", dst_ip_col="IPV4_DST_ADDR",
        src_port_col="L4_SRC_PORT", dst_port_col="L4_DST_PORT",
    )
    source = AnchorBinGraphSource(
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

    model = ArgusModel(cfg, f_e=len(feature_names), f_v=18,
                       class_names=class_names).to(device)
    if ckpt_path is not None and Path(ckpt_path).is_file():
        load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    for bin_id in source.unique_bins:
        batch = source.build_bin_batch(bin_id, f_v=model.f_v)
        if batch is None or batch["n_targets"] == 0:
            continue
        inputs = model_inputs_from_batch(batch, device)
        with torch.no_grad():
            outputs = model(*inputs)
        probs = outputs.get("p_hat", torch.softmax(outputs["logits"], dim=-1))
        preds = probs.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch["target_labels"].numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    report = closed_set_report(y_true, y_pred, class_names, y_score=y_prob)
    tiers = per_tier_macro_f1(report["per_class_f1"], report["support"])
    report["per_tier_macro_f1"] = tiers

    print(f"[06] Closed-set macro-F1: {report['macro_f1']:.4f}")
    print(f"[06] Per-tier F1: {tiers}")
    return report


def run(dataset: str, ckpt_path: str | None = None) -> None:
    report = evaluate_closed_set(dataset, ckpt_path)
    cfg = load_config(dataset=dataset)
    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_closed_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[06] Saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    run(args.dataset, ckpt_path=args.ckpt)
