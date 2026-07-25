"""09 — Cross-dataset transfer evaluation (Protocol C / P-TR).

Train on one dataset, test on another with identical 55-column schemas but
disjoint label vocabularies. Evaluation is by family mapping (dos, ddos, brute,
web, other) plus unknown detection.

This is the ultimate control for host identity leakage (TRAP 3): no host appears
in both datasets, so any transfer performance cannot be identity memorisation.

Usage:
    python scripts/09_eval_transfer.py --source cicids2018 --target ton_iot
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
from argus.constants import FAMILY_OF
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint
from argus.train.loop import model_inputs_from_batch
from argus.eval.openset import open_auc
from argus.eval.metrics import closed_set_report


def _map_to_family(label: str) -> str:
    """Map a canonical label to its attack family, or 'benign'."""
    if label == "benign":
        return "benign"
    return FAMILY_OF.get(label, "other")


def _build_source(df: pd.DataFrame, cfg, feature_names: list[str]) -> AnchorBinGraphSource:
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    src_ids, dst_ids, _ = assign_node_ids(
        df, node_granularity=cfg.graph.node_granularity,
        src_ip_col="IPV4_SRC_ADDR", dst_ip_col="IPV4_DST_ADDR",
        src_port_col="L4_SRC_PORT", dst_port_col="L4_DST_PORT",
    )
    times_ms = df["FLOW_START_MILLISECONDS"].to_numpy()
    edge_features = df[feature_names].to_numpy(dtype=np.float32)
    label_ids = df["_label_id"].to_numpy(dtype=np.int64)
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


def evaluate_transfer(
    source_dataset: str,
    target_dataset: str,
    ckpt_path: str | Path | None = None,
    max_bins: int | None = None,
) -> dict:
    """Evaluate a model trained on source_dataset on target_dataset test data."""
    src_cfg = load_config(dataset=source_dataset)
    tgt_cfg = load_config(dataset=target_dataset)
    device = torch.device(src_cfg.run.device if torch.cuda.is_available() else "cpu")

    # Source artifacts
    src_artifact_dir = resolved_path(src_cfg, "artifact_dir") / source_dataset
    with open(src_artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    with open(src_artifact_dir / "class_vocab.json") as f:
        source_class_names = json.load(f)

    # Target data
    tgt_processed_dir = resolved_path(tgt_cfg, "processed_dir") / target_dataset
    test_parquet = tgt_processed_dir / "test_features.parquet"
    if not test_parquet.is_file():
        print(f"[09] Target test features not found at {test_parquet}")
        print(f"[09] Run 01-03 scripts on {target_dataset} first")
        return {"status": "missing_target_test_features"}

    test_df = pd.read_parquet(test_parquet)

    # Load source model
    model = ArgusModel(src_cfg, f_e=len(feature_names), f_v=18, class_names=source_class_names).to(device)
    if ckpt_path is None:
        default_ckpt = Path(__file__).parents[1] / src_cfg.run.out_dir / f"{source_dataset}_stage2" / "stage2_final.pt"
        if default_ckpt.is_file():
            ckpt_path = default_ckpt
    if ckpt_path is not None and Path(ckpt_path).is_file():
        print(f"[09] Loading checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model, map_location=str(device))

    # Map target labels to families
    target_family_of = {}
    for c in test_df["canonical_label"].unique():
        target_family_of[c] = _map_to_family(c)

    family_set = sorted(set(target_family_of.values()))
    family_to_id = {f: i for i, f in enumerate(family_set)}
    test_df["_family_id"] = test_df["canonical_label"].map(target_family_of).map(family_to_id)
    # Unknown: target class whose family is not in source classes
    source_families = set()
    for c in source_class_names:
        source_families.add(_map_to_family(c))
    test_df["_is_known"] = test_df["canonical_label"].map(target_family_of).isin(source_families)
    test_df["_label_id"] = test_df["_family_id"]  # Use family-level labels for eval

    source = _build_source(test_df, tgt_cfg, feature_names)

    model.eval()
    all_preds, all_targets, all_probs, all_unknown = [], [], [], []
    bins = source.unique_bins[:max_bins] if max_bins else source.unique_bins
    with torch.no_grad():
        for bin_id in bins:
            batch = source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            inputs = model_inputs_from_batch(batch, device)
            outputs = model(*inputs)
            probs = outputs.get("p_hat", torch.softmax(outputs.get("logits", outputs["cos_c"]), dim=-1))
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_targets.append(batch["target_labels"].cpu().numpy())
            all_unknown.append(1.0 - probs.max(dim=-1).values.cpu().numpy())

    if not all_preds:
        return {"status": "no_test_targets"}

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    unknown_scores = np.concatenate(all_unknown)

    # Family-level closed-set report (only on known-family samples)
    known_mask = test_df["_is_known"].values[:len(y_true)]
    if known_mask.sum() > 0:
        family_report = closed_set_report(
            y_true[known_mask], y_pred[known_mask], family_set, y_score=y_probs[known_mask]
        )
    else:
        family_report = {"macro_f1": 0.0, "per_class_f1": {}}

    # Open-set: known families vs unknown
    known_family_ids = {family_to_id[f] for f in source_families if f in family_to_id}
    o_auc = open_auc(y_true, y_probs, known_family_ids)

    # Unknown detection: samples whose family is NOT in source
    truly_unknown = ~known_mask.values[:len(y_true)]
    if truly_unknown.sum() > 0:
        # Using vacuity: high unknown score → predicted unknown
        thresh = np.percentile(unknown_scores, 95)
        pred_unknown = unknown_scores > thresh
        unknown_tpr = float(pred_unknown[truly_unknown].mean())
        unknown_fpr = float(pred_unknown[~truly_unknown].mean()) if (~truly_unknown).sum() > 0 else 0.0
    else:
        unknown_tpr = 0.0
        unknown_fpr = 0.0

    summary = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source_classes": len(source_class_names),
        "family_mapping": {c: _map_to_family(c) for c in source_class_names},
        "target_families": family_set,
        "known_family_macro_f1": family_report.get("macro_f1", 0.0),
        "per_family_f1": family_report.get("per_class_f1", {}),
        "open_auc": o_auc,
        "unknown_tpr": unknown_tpr,
        "unknown_fpr": unknown_fpr,
        "n_target_test": int(len(y_true)),
        "n_known_target": int(known_mask.sum()),
        "n_unknown_target": int(truly_unknown.sum()),
    }
    return summary


def run(source: str, target: str, ckpt: str | None = None, max_bins: int | None = None) -> None:
    results = evaluate_transfer(source, target, ckpt_path=ckpt, max_bins=max_bins)
    cfg = load_config(dataset=source)
    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{source}_to_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transfer_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[09] Family-level macro-F1: {results.get('known_family_macro_f1', 0):.4f}")
    print(f"[09] OpenAUC: {results.get('open_auc', 0):.4f}")
    print(f"[09] Unknown TPR: {results.get('unknown_tpr', 0):.4f}")
    print(f"[09] Saved to {out_dir / 'transfer_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(args.source, args.target, ckpt=args.ckpt, max_bins=args.max_bins)
