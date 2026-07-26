"""07 — Open-set evaluation (Protocol B).

Evaluates unknown detection (OpenAUC, unknown TPR/FPR) over the persisted
Protocol-B holdout sets. Each holdout set is scored with ITS OWN checkpoint —
one trained with those classes genuinely excluded (scripts/02→05 run with
--holdout-index i). Scoring a single all-classes checkpoint and relabelling
classes as UNKNOWN post hoc is NOT open-set evaluation (the model saw every
"held-out" class during training), so this script refuses to run without the
per-holdout artifacts rather than silently producing leaked numbers.

Per-holdout prerequisites (i = 0..4):
    python scripts/02_build_splits.py  --dataset D --protocol B --holdout-index i
    python scripts/03_fit_features.py  --dataset D --holdout-index i
    python scripts/03_cache_graphs.py  --dataset D --holdout-index i   (optional, speeds 04/05)
    python scripts/04_train_encoder.py --dataset D --holdout-index i
    python scripts/05_train_head.py    --dataset D --holdout-index i

Usage:
    python scripts/07_eval_open_set.py --dataset cicids2018
    python scripts/07_eval_open_set.py --dataset cicids2018 --holdout-index 2
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
from argus.eval.openset import open_set_report
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.graph.node_features import node_feature_dim
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint
from argus.train.loop import model_inputs_from_batch


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


def _evaluate_one_holdout(
    cfg, dataset: str, idx: int, holdout_classes: list[str],
    device: torch.device, ckpt_override: str | Path | None,
    max_bins: int | None,
) -> dict | None:
    repo = Path(__file__).parents[1]
    hd_processed = resolved_path(cfg, "processed_dir") / dataset / f"holdout_b{idx}"
    hd_artifacts = resolved_path(cfg, "artifact_dir") / dataset / f"holdout_b{idx}"
    stage1_dir = repo / cfg.run.out_dir / f"{dataset}_stage1_b{idx}"
    ckpt = Path(ckpt_override) if ckpt_override else (
        repo / cfg.run.out_dir / f"{dataset}_stage2_b{idx}" / "stage2_final.pt")

    required = {
        "val_features": hd_processed / "val_features.parquet",
        "test_features": hd_processed / "test_features.parquet",
        "train_features": hd_processed / "train_features.parquet",
        "feature_manifest": hd_artifacts / "feature_manifest.json",
        "class_vocab": stage1_dir / "class_vocab.json",
        "stage2_ckpt": ckpt,
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        print(f"[07]   holdout {idx} {holdout_classes}: SKIPPED — missing {missing} "
              f"(run scripts 02→05 with --holdout-index {idx})")
        return None

    with open(required["feature_manifest"]) as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    with open(required["class_vocab"]) as f:
        class_names = json.load(f)
    overlap = set(class_names) & set(holdout_classes)
    if overlap:
        raise RuntimeError(
            f"holdout {idx}: checkpoint vocab contains held-out classes {sorted(overlap)} — "
            f"this checkpoint was NOT trained holdout-excluded; refusing to report leaked numbers."
        )
    label_to_id = {c: i for i, c in enumerate(class_names)}

    train_counts = pd.read_parquet(
        required["train_features"], columns=["canonical_label"]
    )["canonical_label"].value_counts().to_dict()
    f_v = node_feature_dim(cfg.features.te7_enabled)
    model = ArgusModel(
        cfg, f_e=len(feature_names), f_v=f_v,
        class_names=class_names, class_counts=train_counts,
    ).to(device)
    load_checkpoint(ckpt, model, map_location=str(device))

    def _lbl(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # Protocol-B splits label held-out test rows canonical "UNKNOWN";
        # anything outside the reduced vocab maps to -1.
        df["_label_id"] = df["canonical_label"].map(lambda c: label_to_id.get(c, -1))
        return df

    # Calibrate theta_unknown on validation only (standing rule 6) — the val
    # split is all-known by construction. unknown_score = 1 - max(prob), so a
    # target_false_unknown_rate of 5% needs the 95th percentile of known
    # validation scores.
    val_feat = _lbl(required["val_features"])
    val_source = _build_source(val_feat, cfg, feature_names)
    val_true, _, _, val_unknown_scores = _infer(model, val_source, device, max_bins=max_bins)
    val_known = val_true >= 0
    if val_known.sum() > 0:
        theta = np.percentile(
            val_unknown_scores[val_known], 100 * (1.0 - cfg.head.target_false_unknown_rate))
    else:
        theta = 0.5

    test_feat = _lbl(required["test_features"])
    source = _build_source(test_feat, cfg, feature_names)
    y_true, y_pred, y_scores, unknown_scores = _infer(model, source, device, max_bins=max_bins)
    if len(y_true) == 0:
        print(f"[07]   holdout {idx}: no test targets")
        return None

    known_ids = set(range(len(class_names)))
    result = open_set_report(
        y_true=np.where(y_true >= 0, y_true, -1),
        y_pred=y_pred,
        y_score=y_scores,
        unknown_score=unknown_scores,
        known_class_ids=known_ids,
        class_names=class_names,
        theta_unknown=float(theta),
    )
    result["holdout_index"] = idx
    result["holdout_classes"] = holdout_classes
    result["checkpoint"] = str(ckpt)
    print(f"[07]   holdout {idx} {holdout_classes}: OpenAUC={result['open_auc']:.4f}, "
          f"Unknown TPR={result['unknown_tpr']:.4f}, FPR={result['unknown_fpr']:.4f}")
    return result


def evaluate_open_set(
    dataset: str,
    protocol: str = "B",
    ckpt_path: str | Path | None = None,
    max_bins: int | None = None,
    holdout_index: int | None = None,
) -> dict:
    cfg = load_config(dataset=dataset)
    processed_root = resolved_path(cfg, "processed_dir") / dataset
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    holdout_sets_path = processed_root / "holdout_sets.json"
    if not holdout_sets_path.is_file():
        raise FileNotFoundError(
            f"{holdout_sets_path} not found. Run "
            f"`python scripts/02_build_splits.py --dataset {dataset} --protocol B "
            f"--holdout-index 0` first — it persists the canonical holdout enumeration."
        )
    with open(holdout_sets_path) as f:
        holdout_sets = json.load(f)
    print(f"[07] {len(holdout_sets)} persisted holdout sets from {holdout_sets_path}")

    indices = [holdout_index] if holdout_index is not None else list(range(len(holdout_sets)))
    all_results = []
    for idx in indices:
        r = _evaluate_one_holdout(
            cfg, dataset, idx, holdout_sets[idx], device,
            ckpt_override=ckpt_path if holdout_index is not None else None,
            max_bins=max_bins,
        )
        if r is not None:
            all_results.append(r)

    open_aucs = [r["open_auc"] for r in all_results]
    tprs = [r["unknown_tpr"] for r in all_results]
    fprs = [r["unknown_fpr"] for r in all_results]

    return {
        "protocol": protocol,
        "n_holdout_sets_evaluated": len(all_results),
        "n_holdout_sets_total": len(holdout_sets),
        "open_auc_mean": float(np.mean(open_aucs)) if open_aucs else 0.0,
        "open_auc_std": float(np.std(open_aucs, ddof=1)) if len(open_aucs) > 1 else 0.0,
        "unknown_tpr_mean": float(np.mean(tprs)) if tprs else 0.0,
        "unknown_tpr_std": float(np.std(tprs, ddof=1)) if len(tprs) > 1 else 0.0,
        "unknown_fpr_mean": float(np.mean(fprs)) if fprs else 0.0,
        "unknown_fpr_std": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
        "per_holdout": all_results,
    }


def run(
    dataset: str,
    protocol: str = "B",
    ckpt: str | None = None,
    max_bins: int | None = None,
    holdout_index: int | None = None,
) -> None:
    results = evaluate_open_set(dataset, protocol=protocol, ckpt_path=ckpt,
                                max_bins=max_bins, holdout_index=holdout_index)
    cfg = load_config(dataset=dataset)
    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_open_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "open_set_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[07] OpenAUC (mean ± std over {results['n_holdout_sets_evaluated']} holdouts): "
          f"{results.get('open_auc_mean', 0):.4f} ± {results.get('open_auc_std', 0):.4f}")
    print(f"[07] Saved to {out_dir / 'open_set_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", default="B")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint override — only valid together with --holdout-index")
    parser.add_argument("--max-bins", type=int, default=None)
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Evaluate a single holdout set instead of all")
    args = parser.parse_args()
    run(args.dataset, protocol=args.protocol, ckpt=args.ckpt, max_bins=args.max_bins,
        holdout_index=args.holdout_index)
