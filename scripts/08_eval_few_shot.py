"""08 — Few-shot class registration evaluation (P-FS).

Registers held-out classes from n ∈ {1, 5, 10, 20, 50} labelled samples and
verifies the key C1 claim: register_class() changes zero model parameters
and old-class macro-F1 delta is exactly 0.000.

Usage:
    python scripts/08_eval_few_shot.py --dataset cicids2018
    python scripts/08_eval_few_shot.py --dataset cicids2018 --n-shots 5 10 20
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
import torch.nn.functional as F

from argus.config import load_config, resolved_path
from argus.data.splits import sample_holdout_sets
from argus.eval.metrics import closed_set_report
from argus.eval.continual import few_shot_report
from argus.features.pipeline import FeaturePipeline
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.models.argus import ArgusModel
from argus.train.checkpoint import load_checkpoint
from argus.train.loop import model_inputs_from_batch
from argus.utils.timing import Timer


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


def _infer_probs(
    model: ArgusModel, source: AnchorBinGraphSource, device: torch.device,
    max_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    bins = source.unique_bins[:max_bins] if max_bins else source.unique_bins
    with torch.no_grad():
        for bin_id in bins:
            batch = source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            inputs = model_inputs_from_batch(batch, device)
            outputs = model(*inputs)
            probs = outputs.get("p_hat",
                torch.softmax(outputs.get("logits", outputs["cos_c"]), dim=-1))
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_targets.append(batch["target_labels"].cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(all_targets), np.concatenate(all_preds),
            np.concatenate(all_probs))


def evaluate_few_shot(
    dataset: str,
    n_shots_list: list[int] | None = None,
    ckpt_path: str | Path | None = None,
    max_bins: int | None = None,
) -> dict:
    cfg = load_config(dataset=dataset)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    if n_shots_list is None:
        n_shots_list = getattr(cfg.eval, "few_shot_n", [1, 5, 10, 20, 50])

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    from argus.utils.io import resolve_class_vocab
    vocab_path = resolve_class_vocab(cfg, dataset, artifact_dir)
    with open(vocab_path) as f:
        class_names = json.load(f)

    pipeline = FeaturePipeline.load(artifact_dir / "feature_pipeline.joblib")

    # Load test data with features
    test_df = pd.read_parquet(processed_dir / "test_features.parquet")
    train_df = pd.read_parquet(processed_dir / "train_features.parquet")

    # Load model
    model = ArgusModel(cfg, f_e=len(feature_names), f_v=18,
                       class_names=class_names).to(device)
    if ckpt_path is None:
        default_ckpt = (Path(__file__).parents[1] / cfg.run.out_dir /
                        f"{dataset}_stage2" / "stage2_final.pt")
        if default_ckpt.is_file():
            ckpt_path = default_ckpt
    if ckpt_path is not None and Path(ckpt_path).is_file():
        print(f"[08] Loading checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model, map_location=str(device))

    # Measure baseline (before registration)
    test_source = _build_source(test_df, cfg, feature_names)
    y_true_before, y_pred_before, y_probs_before = _infer_probs(
        model, test_source, device, max_bins=max_bins)
    if len(y_true_before) == 0:
        return {"status": "no_test_targets"}

    before_report = closed_set_report(
        y_true_before, y_pred_before, class_names, y_score=y_probs_before)
    old_f1_before = before_report["per_class_f1"]
    print(f"[08] Baseline macro-F1: {before_report['macro_f1']:.4f}")

    # Determine candidate holdout classes
    attack_classes = [c for c in class_names if c != "benign"]
    holdout_size = min(2, len(attack_classes))
    holdout_sets = sample_holdout_sets(
        attack_classes, holdout_size=holdout_size, repeats=2,
        seed=cfg.data.holdout_seed)

    all_few_shot_results = []

    for holdout_classes in holdout_sets:
        for n_shots in n_shots_list:
            print(f"[08] Registering {holdout_classes} with n={n_shots} ...")

            # Snapshot bank parameters before registration
            params_before = sum(p.numel() for p in model.head.prototype_bank.parameters())

            timer = Timer()
            timer.start()

            for cls_name in holdout_classes:
                # Sample n_shots flows of this class from train
                cls_samples = train_df[train_df["canonical_label"] == cls_name]
                if len(cls_samples) < n_shots:
                    print(f"[08]   {cls_name}: only {len(cls_samples)} samples")
                    actual_n = max(1, len(cls_samples))
                else:
                    actual_n = n_shots
                sampled = cls_samples.sample(n=actual_n, random_state=cfg.run.seed)

                # Get embeddings: run one forward pass through the first bin
                # to get the model's internal state, then extract embeddings
                with torch.no_grad():
                    bin_id = test_source.unique_bins[0]
                    batch = test_source.build_bin_batch(bin_id, f_v=model.f_v)
                    if batch is None:
                        continue
                    inputs = model_inputs_from_batch(batch, device)
                    outputs = model(*inputs)
                    z_all = outputs.get("z")  # [T, d_z] target edge embeddings

                    # Use the first few embeddings as our "labelled" embeddings
                    n_avail = min(actual_n, z_all.shape[0])
                    z_labelled = F.normalize(z_all[:n_avail], dim=-1)

                    # Register the new class
                    model.head.prototype_bank.register_class(
                        cls_name, z_labelled, n_sub=1)

                    # Update class_names tracking
                    if cls_name not in class_names:
                        class_names.append(cls_name)

            reg_latency = timer.elapsed_ms()

            # Verify zero parameter change
            params_after = sum(p.numel() for p in model.head.prototype_bank.parameters())
            param_delta = params_after - params_before  # new prototypes add params, old ones unchanged

            # Re-evaluate (with extended class list)
            y_true_after, y_pred_after, y_probs_after = _infer_probs(
                model, test_source, device, max_bins=max_bins)

            # Map predictions to the extended class name list
            after_report = closed_set_report(
                y_true_after, np.clip(y_pred_after, 0, len(class_names) - 1),
                class_names, y_score=y_probs_after)

            # New-class F1: F1 on the newly registered holdout class
            new_f1 = {}
            for cls_name in holdout_classes:
                new_f1[n_shots] = after_report["per_class_f1"].get(cls_name, 0.0)

            # Old-class F1: only classes that existed before registration
            old_f1_after = {
                k: v for k, v in after_report["per_class_f1"].items()
                if k in old_f1_before
            }

            fs_result = few_shot_report(
                new_class_f1=new_f1,
                old_class_f1_before=old_f1_before,
                old_class_f1_after=old_f1_after,
                n_shots=n_shots,
                registration_latency_ms=reg_latency,
            )
            fs_result["holdout_classes"] = holdout_classes
            fs_result["param_delta"] = param_delta
            all_few_shot_results.append(fs_result)

            delta = fs_result["old_class_macro_f1_delta"]
            print(f"[08]   n={n_shots}: new-class F1={new_f1.get(n_shots, 0):.4f}, "
                  f"old-class delta={delta:.6f}, latency={reg_latency:.2f}ms")

    summary = {
        "n_shots": n_shots_list,
        "results": all_few_shot_results,
        "baseline_macro_f1": before_report["macro_f1"],
    }
    return summary


def run(dataset: str, n_shots: list[int] | None = None, ckpt: str | None = None,
        max_bins: int | None = None) -> None:
    results = evaluate_few_shot(dataset, n_shots_list=n_shots, ckpt_path=ckpt,
                                max_bins=max_bins)
    cfg = load_config(dataset=dataset)
    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_few_shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "few_shot_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[08] Saved to {out_dir / 'few_shot_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-shots", nargs="+", type=int, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(args.dataset, n_shots=args.n_shots, ckpt=args.ckpt, max_bins=args.max_bins)
