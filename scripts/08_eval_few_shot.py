"""08 — Few-shot class registration evaluation (P-FS, table T4).

For each Protocol-B holdout set, loads the checkpoint trained with those
classes genuinely excluded (scripts/02→05 --holdout-index i), registers each
held-out class from n ∈ {1, 5, 10, 20, 50} labelled flows, and verifies the
key C1 claims: registration changes ZERO existing parameters (state-dict
diff, not a numel count) and old-class macro-F1 delta is 0.000.

The n labelled flows are the held-out class's chronologically EARLIEST test
rows (the analyst discovers the new attack early); those exact rows are
excluded from post-registration scoring. A fresh checkpoint is loaded per
(holdout, n) configuration so registrations never accumulate across runs.

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
from argus.eval.metrics import closed_set_report
from argus.eval.continual import few_shot_report
from argus.graph.batching import AnchorBinGraphSource
from argus.graph.builder import assign_node_ids
from argus.graph.node_features import node_feature_dim
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
        te7_enabled=cfg.features.te7_enabled,
        spectral_nbins=cfg.features.spectral_nbins,
        spectral_min_flows=cfg.features.spectral_min_flows,
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


def _embed_flows(model: ArgusModel, df: pd.DataFrame, cfg, feature_names: list[str],
                 device: torch.device) -> torch.Tensor:
    """Embed the given flows by running them through the model as their own
    (sparse-context) anchor-bin source. Registration-time context is exactly
    the n labelled flows the analyst has — disclosed simplification."""
    df = df.copy()
    df["_label_id"] = 0
    source = _build_source(df, cfg, feature_names)
    zs = []
    model.eval()
    with torch.no_grad():
        for bin_id in source.unique_bins:
            batch = source.build_bin_batch(bin_id, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            outputs = model(*model_inputs_from_batch(batch, device))
            zs.append(outputs["z"])
    if not zs:
        return torch.zeros(0, 1)
    return F.normalize(torch.cat(zs, dim=0), dim=-1)


def _snapshot_state(model: ArgusModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _zero_change_check(model: ArgusModel, before: dict[str, torch.Tensor],
                       bank_key: str = "head.prototype_bank.bank") -> dict:
    """State-dict diff: every pre-existing tensor must be bitwise unchanged.
    The prototype bank may only GROW (new rows appended); its first rows must
    be bitwise identical to before."""
    after = model.state_dict()
    changed = []
    for k, v_before in before.items():
        v_after = after.get(k)
        if v_after is None:
            changed.append(f"{k} (removed)")
            continue
        if k == bank_key:
            p_before = v_before.shape[0]
            if v_after.shape[0] < p_before or not torch.equal(v_after[:p_before], v_before):
                changed.append(f"{k} (existing rows modified)")
            continue
        if v_after.shape != v_before.shape or not torch.equal(v_after, v_before):
            changed.append(k)
    return {
        "zero_existing_param_change": len(changed) == 0,
        "changed_tensors": changed,
        "bank_rows_added": int(after[bank_key].shape[0] - before[bank_key].shape[0]),
    }


def evaluate_few_shot(
    dataset: str,
    n_shots_list: list[int] | None = None,
    max_bins: int | None = None,
    max_holdout_sets: int = 2,
) -> dict:
    cfg = load_config(dataset=dataset)
    processed_root = resolved_path(cfg, "processed_dir") / dataset
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).parents[1]

    if n_shots_list is None:
        n_shots_list = getattr(cfg.eval, "few_shot_n", [1, 5, 10, 20, 50])

    holdout_sets_path = processed_root / "holdout_sets.json"
    if not holdout_sets_path.is_file():
        raise FileNotFoundError(
            f"{holdout_sets_path} not found. Run 02_build_splits.py --protocol B first."
        )
    with open(holdout_sets_path) as f:
        holdout_sets = json.load(f)
    holdout_sets = holdout_sets[:max_holdout_sets]

    all_few_shot_results = []
    skipped = []

    for idx, holdout_classes in enumerate(holdout_sets):
        hd_processed = processed_root / f"holdout_b{idx}"
        hd_artifacts = resolved_path(cfg, "artifact_dir") / dataset / f"holdout_b{idx}"
        stage1_dir = repo / cfg.run.out_dir / f"{dataset}_stage1_b{idx}"
        ckpt = repo / cfg.run.out_dir / f"{dataset}_stage2_b{idx}" / "stage2_final.pt"
        required = [hd_processed / "test_features.parquet",
                    hd_processed / "train_features.parquet",
                    hd_artifacts / "feature_manifest.json",
                    stage1_dir / "class_vocab.json", ckpt]
        if any(not p.is_file() for p in required):
            print(f"[08] holdout {idx} {holdout_classes}: SKIPPED — run scripts 02→05 "
                  f"with --holdout-index {idx} first")
            skipped.append(idx)
            continue

        with open(hd_artifacts / "feature_manifest.json") as f:
            feature_names = json.load(f)["feature_names"]
        with open(stage1_dir / "class_vocab.json") as f:
            base_class_names = json.load(f)
        label_to_id = {c: i for i, c in enumerate(base_class_names)}
        train_counts = pd.read_parquet(
            hd_processed / "train_features.parquet", columns=["canonical_label"]
        )["canonical_label"].value_counts().to_dict()
        f_v = node_feature_dim(cfg.features.te7_enabled)

        test_df = pd.read_parquet(hd_processed / "test_features.parquet")
        if "label_pre_holdout" not in test_df.columns:
            print(f"[08] holdout {idx}: test_features lacks label_pre_holdout — rebuild "
                  f"splits/features with current scripts (02/03) first; skipping")
            skipped.append(idx)
            continue

        def _fresh_model() -> ArgusModel:
            m = ArgusModel(cfg, f_e=len(feature_names), f_v=f_v,
                           class_names=list(base_class_names),
                           class_counts=train_counts).to(device)
            load_checkpoint(ckpt, m, map_location=str(device))
            return m

        # Baseline (before registration): known-only rows scored against the
        # reduced vocab.
        known_mask = test_df["canonical_label"] != "UNKNOWN"
        known_df = test_df[known_mask].copy()
        known_df["_label_id"] = known_df["canonical_label"].map(label_to_id)
        model0 = _fresh_model()
        base_source = _build_source(known_df, cfg, feature_names)
        y_true_b, y_pred_b, y_probs_b = _infer_probs(model0, base_source, device, max_bins=max_bins)
        if len(y_true_b) == 0:
            skipped.append(idx)
            continue
        before_report = closed_set_report(y_true_b, y_pred_b, base_class_names, y_score=y_probs_b)
        old_f1_before = before_report["per_class_f1"]
        print(f"[08] holdout {idx} baseline (known-only) macro-F1: {before_report['macro_f1']:.4f}")

        for n_shots in n_shots_list:
            model = _fresh_model()
            class_names = list(base_class_names)
            state_before = _snapshot_state(model)

            shot_row_ids: list[int] = []
            timer = Timer()
            timer.start()
            for cls_name in holdout_classes:
                cls_rows = test_df[test_df["label_pre_holdout"] == cls_name]
                if len(cls_rows) == 0:
                    continue
                shots = cls_rows.nsmallest(min(n_shots, len(cls_rows)),
                                           "FLOW_START_MILLISECONDS")
                shot_row_ids.extend(shots.index.tolist())
                z = _embed_flows(model, shots, cfg, feature_names, device)
                if z.shape[0] == 0:
                    continue
                model.head.prototype_bank.register_class(cls_name, z.to(device), n_sub=1)
                if cls_name not in class_names:
                    class_names.append(cls_name)
            reg_latency = timer.elapsed_ms()

            change = _zero_change_check(model, state_before)

            # Post-registration scoring: everything except the shot rows.
            eval_df = test_df.drop(index=shot_row_ids).copy()
            ext_label_to_id = {c: i for i, c in enumerate(class_names)}
            eval_df["_label_id"] = eval_df["label_pre_holdout"].map(
                lambda c: ext_label_to_id.get(c, -1))
            eval_df = eval_df[eval_df["_label_id"] >= 0]
            eval_source = _build_source(eval_df, cfg, feature_names)
            y_true_a, y_pred_a, y_probs_a = _infer_probs(model, eval_source, device,
                                                         max_bins=max_bins)
            if len(y_true_a) == 0:
                continue
            after_report = closed_set_report(y_true_a, y_pred_a, class_names,
                                             y_score=y_probs_a)

            new_f1 = {n_shots: float(np.mean([
                after_report["per_class_f1"].get(c, 0.0) for c in holdout_classes
            ]))}
            old_f1_after = {k: v for k, v in after_report["per_class_f1"].items()
                            if k in old_f1_before}

            fs_result = few_shot_report(
                new_class_f1=new_f1,
                old_class_f1_before=old_f1_before,
                old_class_f1_after=old_f1_after,
                n_shots=n_shots,
                registration_latency_ms=reg_latency,
            )
            fs_result["holdout_index"] = idx
            fs_result["holdout_classes"] = holdout_classes
            fs_result["per_new_class_f1"] = {
                c: after_report["per_class_f1"].get(c, 0.0) for c in holdout_classes}
            fs_result.update(change)
            all_few_shot_results.append(fs_result)

            print(f"[08]   n={n_shots}: new-class mean F1={new_f1[n_shots]:.4f}, "
                  f"old-class delta={fs_result['old_class_macro_f1_delta']:.6f}, "
                  f"zero-change={change['zero_existing_param_change']}, "
                  f"latency={reg_latency:.2f}ms")

    return {
        "n_shots": n_shots_list,
        "results": all_few_shot_results,
        "skipped_holdout_indices": skipped,
    }


def run(dataset: str, n_shots: list[int] | None = None,
        max_bins: int | None = None) -> None:
    results = evaluate_few_shot(dataset, n_shots_list=n_shots, max_bins=max_bins)
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
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(args.dataset, n_shots=args.n_shots, max_bins=args.max_bins)
