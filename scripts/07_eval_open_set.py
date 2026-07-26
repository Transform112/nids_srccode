"""07 — Open-set evaluation (Protocol B / B2).

Holds out attack classes, evaluates unknown detection, computes OpenAUC and
unknown TPR/FPR. Runs over 5 distinct holdout sets; reports mean ± std.

Requires a trained Stage-2 checkpoint (scripts/04 + 05) and Protocol-B splits
from 02_build_splits.py.

Usage:
    python scripts/07_eval_open_set.py --dataset cicids2018
    python scripts/07_eval_open_set.py --dataset cicids2018 --protocol B2 --ckpt path/to/stage2_final.pt
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
from argus.data.splits import protocol_b_split, sample_holdout_sets
from argus.eval.openset import open_set_report
from argus.eval.metrics import closed_set_report
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint
from argus.train.loop import model_inputs_from_batch
from argus.features.pipeline import FeaturePipeline


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


def _infer(
    model: ArgusModel, source: AnchorBinGraphSource, device: torch.device, max_bins: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred, scores, unknown_scores)."""
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
            # Unknown score: 1 - max probability (vacuity proxy)
            all_unknown.append(1.0 - probs.max(dim=-1).values.cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_unknown),
    )


def evaluate_open_set(
    dataset: str,
    protocol: str = "B",
    ckpt_path: str | Path | None = None,
    max_bins: int | None = None,
) -> dict:
    cfg = load_config(dataset=dataset)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    # Load artifacts
    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]

    from argus.utils.io import resolve_class_vocab
    vocab_path = resolve_class_vocab(cfg, dataset, artifact_dir)
    with open(vocab_path) as f:
        class_names = json.load(f)

    pipeline = FeaturePipeline.load(artifact_dir / "feature_pipeline.joblib")

    # Load test data (Protocol-A split already processed)
    test_parquet = processed_dir / "test.parquet"
    if not test_parquet.is_file():
        print(f"[07] Test split not found at {test_parquet}; run 02_build_splits.py first")
        return {"status": "missing_test_split"}

    test_df = pd.read_parquet(test_parquet)

    # Determine attack classes (all non-benign)
    attack_classes = [c for c in class_names if c != "benign"]
    holdout_size = min(cfg.data.holdout_size, len(attack_classes))
    holdout_sets = sample_holdout_sets(
        attack_classes, holdout_size=holdout_size,
        repeats=cfg.data.holdout_repeats, seed=cfg.data.holdout_seed,
    )
    print(f"[07] {len(holdout_sets)} holdout sets of size {holdout_size}:")
    for hs in holdout_sets:
        print(f"[07]   {hs}")

    # Load model
    model = ArgusModel(cfg, f_e=len(feature_names), f_v=18, class_names=class_names).to(device)
    if ckpt_path is None:
        default_ckpt = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_stage2" / "stage2_final.pt"
        if default_ckpt.is_file():
            ckpt_path = default_ckpt
    if ckpt_path is not None and Path(ckpt_path).is_file():
        print(f"[07] Loading checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model, map_location=str(device))

    # Run per-holdout evaluation
    all_results = []
    for holdout_classes in holdout_sets:
        # Build Protocol-B test: mark holdout classes as UNKNOWN
        holdout_mask = test_df["canonical_label"].isin(holdout_classes)
        test_b = test_df.copy()
        test_b["_label_id"] = test_b["canonical_label"].apply(
            lambda c: -1 if c in holdout_classes else class_names.index(c)
            if c in class_names else -1
        )
        # Known class IDs: all class_names except holdout classes
        known_ids = {i for i, c in enumerate(class_names) if c not in holdout_classes}

        # Transform features
        test_feat = pipeline.transform(test_b)
        for extra in ("FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                       "L4_SRC_PORT", "L4_DST_PORT"):
            if extra in test_b.columns:
                test_feat[extra] = test_b[extra].values
        test_feat["_label_id"] = test_b["_label_id"].values

        source = _build_source(test_feat, cfg, feature_names)
        y_true, y_pred, y_scores, unknown_scores = _infer(model, source, device, max_bins=max_bins)

        if len(y_true) == 0:
            continue

        # Map -1 (UNKNOWN) to last column for score matrix
        n_known = len(class_names)
        score_matrix = np.zeros((len(y_true), n_known), dtype=np.float32)
        known_pred_mask = y_pred < n_known
        for i in range(len(y_true)):
            if known_pred_mask[i]:
                score_matrix[i, y_pred[i]] = 1.0

        theta = np.percentile(unknown_scores, 100 * cfg.head.target_false_unknown_rate)

        result = open_set_report(
            y_true=np.where(y_true >= 0, y_true, -1),
            y_pred=y_pred,
            y_score=score_matrix,
            unknown_score=unknown_scores,
            known_class_ids=known_ids,
            class_names=class_names,
            theta_unknown=float(theta),
        )
        result["holdout_classes"] = holdout_classes
        all_results.append(result)
        print(f"[07]   {holdout_classes}: OpenAUC={result['open_auc']:.4f}, "
              f"Unknown TPR={result['unknown_tpr']:.4f}, FPR={result['unknown_fpr']:.4f}")

    # Aggregate
    open_aucs = [r["open_auc"] for r in all_results]
    tprs = [r["unknown_tpr"] for r in all_results]
    fprs = [r["unknown_fpr"] for r in all_results]

    summary = {
        "protocol": protocol,
        "n_holdout_sets": len(all_results),
        "holdout_size": holdout_size,
        "open_auc_mean": float(np.mean(open_aucs)) if open_aucs else 0.0,
        "open_auc_std": float(np.std(open_aucs, ddof=1)) if len(open_aucs) > 1 else 0.0,
        "unknown_tpr_mean": float(np.mean(tprs)) if tprs else 0.0,
        "unknown_tpr_std": float(np.std(tprs, ddof=1)) if len(tprs) > 1 else 0.0,
        "unknown_fpr_mean": float(np.mean(fprs)) if fprs else 0.0,
        "unknown_fpr_std": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
        "per_holdout": all_results,
    }
    return summary


def run(
    dataset: str,
    protocol: str = "B",
    ckpt: str | None = None,
    max_bins: int | None = None,
) -> None:
    results = evaluate_open_set(dataset, protocol=protocol, ckpt_path=ckpt, max_bins=max_bins)
    cfg = load_config(dataset=dataset)
    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_open_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "open_set_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[07] OpenAUC (mean ± std): {results.get('open_auc_mean', 0):.4f} "
          f"± {results.get('open_auc_std', 0):.4f}")
    print(f"[07] Saved to {out_dir / 'open_set_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", default="B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(args.dataset, protocol=args.protocol, ckpt=args.ckpt, max_bins=args.max_bins)
