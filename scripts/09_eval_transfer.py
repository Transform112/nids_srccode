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
from argus.graph.node_features import node_feature_dim
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
        te7_enabled=cfg.features.te7_enabled,
        spectral_nbins=cfg.features.spectral_nbins,
        spectral_min_flows=cfg.features.spectral_min_flows,
    )


def _run_inference(
    source: AnchorBinGraphSource, model: ArgusModel, device: torch.device, max_bins: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, y_probs, unknown_scores) over all bins."""
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
    if not all_targets:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(all_targets), np.concatenate(all_preds),
        np.concatenate(all_probs), np.concatenate(all_unknown),
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

    from argus.utils.io import resolve_class_vocab
    vocab_path = resolve_class_vocab(src_cfg, source_dataset, src_artifact_dir)
    with open(vocab_path) as f:
        source_class_names = json.load(f)

    # Target data
    tgt_processed_dir = resolved_path(tgt_cfg, "processed_dir") / target_dataset
    test_parquet = tgt_processed_dir / "test_features.parquet"
    val_parquet = tgt_processed_dir / "val_features.parquet"
    if not test_parquet.is_file():
        print(f"[09] Target test features not found at {test_parquet}")
        print(f"[09] Run 01-03 scripts on {target_dataset} first")
        return {"status": "missing_target_test_features"}
    if not val_parquet.is_file():
        print(f"[09] Target val features not found at {val_parquet}")
        print(f"[09] Run 01-03 scripts on {target_dataset} first")
        return {"status": "missing_target_val_features"}

    test_df = pd.read_parquet(test_parquet)
    val_df = pd.read_parquet(val_parquet)

    # Load source model — class_counts must match what the checkpoint was
    # trained with (PrototypeBank's sub-prototype count per class depends on it).
    src_processed_dir = resolved_path(src_cfg, "processed_dir") / source_dataset
    src_train_counts = pd.read_parquet(
        src_processed_dir / "train_features.parquet", columns=["canonical_label"]
    )
    class_counts = src_train_counts["canonical_label"].value_counts().to_dict()
    f_v = node_feature_dim(src_cfg.features.te7_enabled)
    model = ArgusModel(
        src_cfg, f_e=len(feature_names), f_v=f_v,
        class_names=source_class_names, class_counts=class_counts,
    ).to(device)
    if ckpt_path is None:
        default_ckpt = Path(__file__).parents[1] / src_cfg.run.out_dir / f"{source_dataset}_stage2" / "stage2_final.pt"
        if default_ckpt.is_file():
            ckpt_path = default_ckpt
    if ckpt_path is not None and Path(ckpt_path).is_file():
        print(f"[09] Loading checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model, map_location=str(device))

    # Map target labels to families (fit the mapping on test; val reuses it —
    # target_dataset's family vocabulary is a fixed property of the dataset).
    target_family_of = {}
    for c in test_df["canonical_label"].unique():
        target_family_of[c] = _map_to_family(c)
    for c in val_df["canonical_label"].unique():
        target_family_of.setdefault(c, _map_to_family(c))

    family_set = sorted(set(target_family_of.values()))
    family_to_id = {f: i for i, f in enumerate(family_set)}
    source_families = {_map_to_family(c) for c in source_class_names}
    known_family_ids = {family_to_id[f] for f in source_families if f in family_to_id}

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_family_id"] = df["canonical_label"].map(target_family_of).map(family_to_id)
        df["_label_id"] = df["_family_id"]  # family-level labels for eval
        return df

    # Calibrate the unknown threshold on target validation only (standing
    # rule 6) — never on the test split scored below. known/unknown status is
    # derived from the returned family ids (aligned with the inference output
    # order), not by re-indexing into the input dataframe by position —
    # _build_source sorts rows by timestamp internally, so a positional slice
    # of the original (unsorted) dataframe would silently misalign.
    val_df = _prepare(val_df)
    val_source = _build_source(val_df, tgt_cfg, feature_names)
    val_true, _, _, val_unknown_scores = _run_inference(val_source, model, device, max_bins=max_bins)
    val_known_mask = np.isin(val_true, list(known_family_ids))
    if val_known_mask.sum() > 0:
        thresh = np.percentile(val_unknown_scores[val_known_mask], 95)
    else:
        thresh = 0.5

    test_df = _prepare(test_df)
    source = _build_source(test_df, tgt_cfg, feature_names)
    y_true, y_pred, y_probs, unknown_scores = _run_inference(source, model, device, max_bins=max_bins)

    if len(y_true) == 0:
        return {"status": "no_test_targets"}

    # Family-level closed-set report (only on known-family samples)
    known_mask = np.isin(y_true, list(known_family_ids))
    if known_mask.sum() > 0:
        family_report = closed_set_report(
            y_true[known_mask], y_pred[known_mask], family_set, y_score=y_probs[known_mask]
        )
    else:
        family_report = {"macro_f1": 0.0, "per_class_f1": {}}

    # Open-set: known families vs unknown
    o_auc = open_auc(y_true, y_probs, known_family_ids)

    # Unknown detection: samples whose family is NOT in source
    truly_unknown = ~known_mask
    if truly_unknown.sum() > 0:
        # Using vacuity: high unknown score → predicted unknown. thresh was
        # calibrated on target validation above, not on this test split.
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
