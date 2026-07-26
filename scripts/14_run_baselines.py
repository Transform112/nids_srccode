"""14 — Run baselines on the same Protocol-A splits ARGUS uses.

Baselines (docs/05_ARCHITECTURE.md §8): Extra Trees, Random Forest, MLP
(flow-independent), Identity-only (leakage floor), E-GraphSAGE and EGATv2
(GNN bars).

Usage:
    python scripts/14_run_baselines.py --dataset cicids2018
    python scripts/14_run_baselines.py --dataset cicids2018 --skip-egraphsage --skip-egatv2
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

from argus.config import load_config, resolved_path  # noqa: E402
from argus.eval.metrics import closed_set_report  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids  # noqa: E402
from argus.models.baselines.egatv2 import EGATv2  # noqa: E402
from argus.models.baselines.egraphsage import EGraphSAGE  # noqa: E402
from argus.models.baselines.identity_only import IdentityOnlyClassifier  # noqa: E402
from argus.models.baselines.tabular import TabularBaseline  # noqa: E402
from argus.utils.io import derive_class_vocab  # noqa: E402


def _run_tabular(name: str, baseline, x_train, y_train, x_test, y_test, class_names) -> dict:
    print(f"[14] Training {name} ...")
    baseline.fit(x_train, y_train)
    pred = baseline.predict(x_test)
    return closed_set_report(y_test, pred, class_names)


def _run_identity_only(train_df, test_df, y_train, y_test, class_names) -> dict:
    print("[14] Training identity-only (leakage floor) ...")
    clf = IdentityOnlyClassifier(max_depth=8, seed=0)
    clf.fit(train_df, y_train, test_df=test_df)
    pred = clf.predict(test_df)
    return closed_set_report(y_test, pred, class_names)


def _gnn_baseline_source(df: pd.DataFrame, cfg, feature_names: list[str]) -> AnchorBinGraphSource:
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
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
        neighbour_cap=10**6,  # "uncapped" neighbourhoods per docs/05 §8
        sampling="uniform",
    )


def _run_gnn_baseline(
    name, model, train_df, test_df, cfg, feature_names, class_names, device, max_bins=None
) -> dict:
    print(f"[14] Training {name} (mid scale, uncapped neighbourhoods) ...")
    train_source = _gnn_baseline_source(train_df, cfg, feature_names)
    test_source = _gnn_baseline_source(test_df, cfg, feature_names)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    bins = train_source.unique_bins[:max_bins] if max_bins else train_source.unique_bins
    model.train()
    for epoch in range(3):
        total_loss, n = 0.0, 0
        for b in bins:
            batch = train_source.build_bin_batch(b, f_v=1)
            if batch is None or batch["n_targets"] == 0:
                continue
            ei_m, ea_m, _ = batch["scale_mid"]
            n_nodes = batch["node_feat"].shape[0]
            target_edge_index = batch["target_edge_index"].to(device)
            target_edge_attr = batch["target_edge_attr"].to(device)
            targets = batch["target_labels"].to(device)
            opt.zero_grad()
            logits = model(n_nodes, ei_m.to(device), ea_m.to(device), target_edge_index, target_edge_attr)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
        print(f"[14]   epoch {epoch}: loss={total_loss / max(n, 1):.4f} ({n} bins)")

    model.eval()
    all_preds, all_targets = [], []
    test_bins = test_source.unique_bins[:max_bins] if max_bins else test_source.unique_bins
    with torch.no_grad():
        for b in test_bins:
            batch = test_source.build_bin_batch(b, f_v=1)
            if batch is None or batch["n_targets"] == 0:
                continue
            ei_m, ea_m, _ = batch["scale_mid"]
            n_nodes = batch["node_feat"].shape[0]
            target_edge_index = batch["target_edge_index"].to(device)
            target_edge_attr = batch["target_edge_attr"].to(device)
            logits = model(n_nodes, ei_m.to(device), ea_m.to(device), target_edge_index, target_edge_attr)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(batch["target_labels"].numpy())

    if not all_preds:
        return {"error": "no test bins produced targets"}
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return closed_set_report(y_true, y_pred, class_names)


def run(
    dataset: str,
    skip_egraphsage: bool = False,
    skip_egatv2: bool = False,
    max_bins: int | None = None,
) -> dict:
    cfg = load_config(dataset=dataset)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]

    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    test_df = pd.read_parquet(processed_dir / "test_features.parquet")

    # Baselines train their own models from scratch (no ARGUS checkpoint to
    # match vocab against), so derive directly from data rather than
    # reading/writing a class_vocab.json — artifact_dir may be a read-only
    # Kaggle input mount, and this script may legitimately run before any
    # ARGUS training (AGENT_GUIDE.md: "P2 is deliberately early").
    class_names = derive_class_vocab(train_df)
    label_to_id = {c: i for i, c in enumerate(class_names)}

    # _label_id stays in-memory only — processed_dir may be a read-only
    # Kaggle input mount, so never write derived columns back to it.
    train_df["_label_id"] = train_df["canonical_label"].map(label_to_id)
    test_df["_label_id"] = test_df["canonical_label"].map(label_to_id)

    y_train = train_df["_label_id"].to_numpy()
    y_test = test_df["_label_id"].to_numpy()
    x_train = train_df[feature_names].to_numpy(dtype=np.float32)
    x_test = test_df[feature_names].to_numpy(dtype=np.float32)

    results: dict[str, dict] = {}

    results["extra_trees"] = _run_tabular(
        "Extra Trees", TabularBaseline.extra_trees(n_estimators=100),
        x_train, y_train, x_test, y_test, class_names,
    )
    results["random_forest"] = _run_tabular(
        "Random Forest", TabularBaseline.random_forest(n_estimators=100),
        x_train, y_train, x_test, y_test, class_names,
    )
    results["mlp"] = _run_tabular(
        "MLP", TabularBaseline.mlp(), x_train, y_train, x_test, y_test, class_names,
    )
    results["identity_only"] = _run_identity_only(train_df, test_df, y_train, y_test, class_names)

    if not skip_egraphsage or not skip_egatv2:
        device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")

    if not skip_egraphsage:
        model = EGraphSAGE(f_e=f_e, d_h=64, num_classes=len(class_names), layers=2)
        results["egraphsage"] = _run_gnn_baseline(
            "E-GraphSAGE", model, train_df, test_df, cfg, feature_names, class_names, device, max_bins=max_bins
        )

    if not skip_egatv2:
        model = EGATv2(f_e=f_e, d_h=64, num_classes=len(class_names), layers=2, heads=4)
        results["egatv2"] = _run_gnn_baseline(
            "EGATv2", model, train_df, test_df, cfg, feature_names, class_names, device, max_bins=max_bins
        )

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n[14] Summary (macro-F1):")
    for name, r in results.items():
        if "macro_f1" in r:
            print(f"[14]   {name:15s} macro_f1={r['macro_f1']:.4f}  identity_floor_check={'*' if name=='identity_only' else ''}")
    print(f"[14] Wrote {out_dir / 'baseline_results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skip-egraphsage", action="store_true")
    parser.add_argument("--skip-egatv2", action="store_true")
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(
        args.dataset,
        skip_egraphsage=args.skip_egraphsage,
        skip_egatv2=args.skip_egatv2,
        max_bins=args.max_bins,
    )
