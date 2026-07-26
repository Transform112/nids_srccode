"""12 — Run the XAI evaluation: native attribution, baseline explainers,
explanation-quality metrics, and UNKNOWN triage.

Requires a trained checkpoint (scripts/04 + 05) and a fitted feature pipeline
(scripts/03). Samples a handful of correctly-detected malicious flows and
UNKNOWN-verdicted flows from the test split.

Usage:
    python scripts/12_run_xai.py --dataset cicids2018 --n-samples 10
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

from argus.config import load_config, resolved_path  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.checkpoint import load_checkpoint  # noqa: E402
from argus.train.loop import model_inputs_from_batch  # noqa: E402
from argus.xai.evidence_attrib import embedding_attribution, integrated_gradients_feature_attribution  # noqa: E402
from argus.xai.explainers import (  # noqa: E402
    attention_weights_baseline,
    gnnexplainer_edge_mask,
    kernelshap_feature_importance,
)
from argus.xai.metrics import feature_fidelity  # noqa: E402
from argus.xai.triage import render_triage_report, validate_unknown_clusters  # noqa: E402


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


def run(dataset: str, n_samples: int = 10, overrides: list[str] | None = None, seed: int = 0) -> dict:
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset
    artifact_dir = resolved_path(cfg, "artifact_dir") / dataset

    with open(artifact_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_names = manifest["feature_names"]
    f_e = manifest["f_e"]

    from argus.utils.io import resolve_class_vocab
    vocab_path = resolve_class_vocab(cfg, dataset, artifact_dir)
    with open(vocab_path) as f:
        class_names = json.load(f)
    label_to_id = {c: i for i, c in enumerate(class_names)}

    feat_test = pd.read_parquet(processed_dir / "test_features.parquet")
    feat_test = feat_test.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    feat_test["_label_id"] = feat_test["canonical_label"].map(label_to_id)

    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    model = ArgusModel(cfg, f_e=f_e, f_v=18, class_names=class_names).to(device)

    ckpt_dir = Path(__file__).parents[1] / cfg.run.out_dir
    stage2_ckpt = ckpt_dir / f"{dataset}_stage2" / "stage2_final.pt"
    if not stage2_ckpt.exists():
        raise FileNotFoundError("Run scripts/04 and 05 first to produce a Stage-2 checkpoint.")
    load_checkpoint(stage2_ckpt, model)
    model.eval()

    source = _build_source(feat_test, cfg, feature_names)
    benign_median = torch.as_tensor(
        feat_test.loc[feat_test["canonical_label"] == cfg.classes.benign, feature_names].median().to_numpy(),
        dtype=torch.float32,
    )

    rng = np.random.default_rng(seed)
    attack_idxs = np.nonzero(feat_test["canonical_label"].to_numpy() != cfg.classes.benign)[0]
    sample_idxs = rng.choice(attack_idxs, size=min(n_samples, len(attack_idxs)), replace=False)

    results: dict[str, list] = {"native_attribution": [], "explainer_metrics": []}
    unknown_embeddings, unknown_true_labels = [], []

    for row_pos in sample_idxs:
        bin_id = int(source.bin_ids[row_pos])
        ranges = dict((b, (lo, hi)) for b, lo, hi in source.ranges["short"])
        if bin_id not in ranges:
            continue
        lo, hi = ranges[bin_id]
        target_positions = np.nonzero(source.bin_ids[lo:hi] == bin_id)[0] + lo
        matches = np.nonzero(target_positions == row_pos)[0]
        if len(matches) == 0:
            continue
        target_idx = int(matches[0])

        batch = source.build_bin_batch(bin_id, f_v=model.f_v)
        if batch is None or batch["n_targets"] == 0:
            continue

        with torch.no_grad():
            outputs = model(*model_inputs_from_batch(batch, device))
        pred_class = int(outputs["p_hat"][target_idx].argmax().item())
        decisions, _ = model.head.decide(outputs) if hasattr(model.head, "decide") else (None, None)
        verdict = int(decisions[target_idx].item()) if decisions is not None else pred_class

        if verdict == -1:  # UNKNOWN
            unknown_embeddings.append(outputs["z"][target_idx].detach().numpy())
            unknown_true_labels.append(int(feat_test["_label_id"].iloc[row_pos]))
            gate = None
            report = render_triage_report(model.head, outputs, target_idx, class_names, gate=gate)
            results.setdefault("triage_reports", []).append(asdict(report))
            continue

        # Native attribution (embedding-level, exact).
        proto_idx_mask = torch.tensor(
            [i == pred_class for i in model.head.prototype_bank.class_of]
        )
        prototype = model.head.prototype_bank.bank.data[proto_idx_mask][0]
        embed_attr = embedding_attribution(outputs["z"][target_idx], prototype, model.head.tau)

        def forward_log_e(edge_attr_1: torch.Tensor) -> torch.Tensor:
            local_batch = dict(batch)
            tea = batch["target_edge_attr"].clone()
            tea[target_idx] = edge_attr_1[0]
            local_batch["target_edge_attr"] = tea
            out = model(*model_inputs_from_batch(local_batch, device))
            return out["log_e"]

        ig_attr, f_x, f_base = integrated_gradients_feature_attribution(
            forward_log_e, batch["target_edge_attr"][target_idx], benign_median, class_idx=pred_class, steps=20,
        )
        results["native_attribution"].append({
            "row": int(row_pos), "pred_class": class_names[pred_class],
            "ig_completeness_gap": abs(float(ig_attr.sum().item()) - (f_x - f_base)),
            "top_features": [feature_names[i] for i in torch.argsort(-ig_attr.abs())[:15].tolist()],
        })

        # Baseline explainers + fidelity metrics (bounded epochs for feasibility).
        attn = attention_weights_baseline(model, batch, device)
        mask_result = gnnexplainer_edge_mask(model, batch, device, target_idx, pred_class, epochs=30)
        shap_attr = kernelshap_feature_importance(
            model, batch, device, target_idx, pred_class, benign_median, n_samples=cfg.xai.shap_nsamples, seed=seed,
        )
        fidelity = feature_fidelity(
            model, batch, device, target_idx, pred_class,
            attribution=ig_attr.detach().numpy(), baseline=benign_median, k=cfg.xai.topk_features,
        )
        results["explainer_metrics"].append({
            "row": int(row_pos), "pred_class": class_names[pred_class],
            "gnnexplainer_final_loss": mask_result.final_loss,
            "attention_nodes_recorded": len(attn),
            "shap_top_features": [feature_names[i] for i in np.argsort(-np.abs(shap_attr))[:15].tolist()],
            "fidelity_plus": fidelity.fidelity_plus, "fidelity_minus": fidelity.fidelity_minus,
            "sparsity": fidelity.sparsity,
        })

    if unknown_embeddings:
        n_true_classes = len(set(unknown_true_labels))
        cluster_result = validate_unknown_clusters(
            np.stack(unknown_embeddings), np.array(unknown_true_labels),
            n_clusters=max(n_true_classes, 1), method="kmeans", seed=seed,
        )
        results["unknown_cluster_validation"] = asdict(cluster_result)

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "xai_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[12] {len(results['native_attribution'])} native-attribution decisions, "
          f"{len(results['explainer_metrics'])} baseline-explainer decisions, "
          f"{len(results.get('triage_reports', []))} UNKNOWN triage reports")
    print(f"[12] Wrote {out_dir / 'xai_results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, n_samples=args.n_samples, overrides=args.overrides, seed=args.seed)
