"""10 — Temporal ablation ladder L0->L7 (docs/09_TEMPORAL_STUDY.md §3).

Cumulative rungs, encoder-only (Stage-1 nearest-prototype classification), so
this does not depend on the (higher-risk) evidential head — it can run before
Stage 2 exists and de-risks C3 independently of C1.

Rung -> flags:
    L0: no temporal columns at all (features.include_temporal_block=False), single scale
    L1: + raw NF-v3 temporal columns (te1=off, te2=off, include_temporal_block=True)
    L2: + TE1 heavy-tail conditioning (te1=on)
    L3: + TE2 derived rhythm descriptors (te2=on)
    L4: + TE3 Time2Vec Delta-t encoding
    L5: + TE4 time-decayed attention
    L6: + TE5 multi-scale windows + TE6 node memory
    L7: + TE7 spectral descriptor (= full ARGUS)

L0-L3 also force `model.te3/4/5/6_enabled=false` and single-scale training
(only the mid scale is used, per docs "L0: single scale (mid), no dt, no
decay, no memory, no spectral") — implemented by only building/training on
the mid-scale context (short/long context tensors are still required by
`ArgusModel.forward`'s signature, so they are filled with the mid-scale
tensors too when `te5_enabled=False`; the multi-scale fusion gate then
degenerates to always selecting the single available scale).

Usage (bounded local dev run):
    python scripts/10_run_temporal_ladder.py --dataset cicids2018 --nrows 50000 \
        --rungs L0 L1 L2 --seeds 0 --epochs 2 --max-bins 100

Usage (Kaggle-scale full sweep, 8 rungs x 5 seeds = 40 runs):
    python scripts/10_run_temporal_ladder.py --dataset cicids2018
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
from argus.features.pipeline import FeaturePipeline  # noqa: E402
from argus.graph.batching import AnchorBinGraphSource  # noqa: E402
from argus.graph.builder import assign_node_ids  # noqa: E402
from argus.models.argus import ArgusModel  # noqa: E402
from argus.train.loop import model_inputs_from_batch  # noqa: E402
from argus.train.stage1_encoder import train_stage1  # noqa: E402

RUNG_FLAGS = {
    "L0": {"include_temporal_block": False, "te1": False, "te2": False,
           "te3": False, "te4": False, "te5": False, "te6": False, "te7": False},
    "L1": {"include_temporal_block": True, "te1": False, "te2": False,
           "te3": False, "te4": False, "te5": False, "te6": False, "te7": False},
    "L2": {"include_temporal_block": True, "te1": True, "te2": False,
           "te3": False, "te4": False, "te5": False, "te6": False, "te7": False},
    "L3": {"include_temporal_block": True, "te1": True, "te2": True,
           "te3": False, "te4": False, "te5": False, "te6": False, "te7": False},
    "L4": {"include_temporal_block": True, "te1": True, "te2": True,
           "te3": True, "te4": False, "te5": False, "te6": False, "te7": False},
    "L5": {"include_temporal_block": True, "te1": True, "te2": True,
           "te3": True, "te4": True, "te5": False, "te6": False, "te7": False},
    "L6": {"include_temporal_block": True, "te1": True, "te2": True,
           "te3": True, "te4": True, "te5": True, "te6": True, "te7": False},
    "L7": {"include_temporal_block": True, "te1": True, "te2": True,
           "te3": True, "te4": True, "te5": True, "te6": True, "te7": True},
}
ALL_RUNGS = list(RUNG_FLAGS.keys())


def _build_source(df: pd.DataFrame, cfg, feature_names: list[str], te7_enabled: bool) -> AnchorBinGraphSource:
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
        te7_enabled=te7_enabled,
    )


def _evaluate_per_class(model: ArgusModel, source: AnchorBinGraphSource, device, class_names, max_bins=None):
    model.eval()
    all_preds, all_targets = [], []
    bins = source.unique_bins[:max_bins] if max_bins else source.unique_bins
    with torch.no_grad():
        for b in bins:
            batch = source.build_bin_batch(b, f_v=model.f_v)
            if batch is None or batch["n_targets"] == 0:
                continue
            outputs = model(*model_inputs_from_batch(batch, device))
            preds = outputs["cos_c"].argmax(dim=1)  # Stage-1 nearest-prototype classification
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["target_labels"].numpy())
    if not all_preds:
        return None
    return closed_set_report(np.concatenate(all_targets), np.concatenate(all_preds), class_names)


def run_one_rung(
    rung: str, dataset: str, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
    class_names: list[str], epochs: int, max_bins: int | None,
) -> dict:
    flags = RUNG_FLAGS[rung]
    overrides = [
        f"model.te3_enabled={str(flags['te3']).lower()}",
        f"model.te4_enabled={str(flags['te4']).lower()}",
        f"model.te5_enabled={str(flags['te5']).lower()}",
        f"model.te6_enabled={str(flags['te6']).lower()}",
        f"train.stage1_epochs={epochs}",
        f"run.seed={seed}",
        "run.device=cpu",
    ]
    cfg = load_config(dataset=dataset, model="argus", overrides=overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = FeaturePipeline(
        protocol_topk=cfg.features.protocol_topk, l7_proto_topk=cfg.features.l7_proto_topk,
        dst_port_topk=cfg.features.dst_port_topk, te1_enabled=flags["te1"], te2_enabled=flags["te2"],
        include_temporal_block=flags["include_temporal_block"],
    )
    pipeline.fit(train_df)
    feature_names = pipeline.feature_names_
    train_feat = pipeline.transform(train_df)
    val_feat = pipeline.transform(val_df)
    for extra in ("canonical_label", "FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                  "L4_SRC_PORT", "L4_DST_PORT"):
        train_feat[extra] = train_df[extra].values
        val_feat[extra] = val_df[extra].values

    label_to_id = {c: i for i, c in enumerate(class_names)}
    train_feat["_label_id"] = train_feat["canonical_label"].map(label_to_id)
    val_feat["_label_id"] = val_feat["canonical_label"].map(label_to_id)

    train_source = _build_source(train_feat, cfg, feature_names, te7_enabled=flags["te7"])
    val_source = _build_source(val_feat, cfg, feature_names, te7_enabled=flags["te7"])

    device = torch.device("cpu")
    class_counts = train_feat["canonical_label"].value_counts().to_dict()
    model = ArgusModel(cfg, f_e=len(feature_names), f_v=18, class_names=class_names, class_counts=class_counts).to(device)

    result = train_stage1(model, train_source, val_source, cfg, device, max_bins=max_bins,
                          class_counts=class_counts)
    report = _evaluate_per_class(model, val_source, device, class_names, max_bins=max_bins)

    return {
        "rung": rung, "seed": seed, "f_e": len(feature_names),
        "best_val_macro_f1": result["best_val_macro_f1"], "history": result["history"],
        "macro_f1": report["macro_f1"] if report else None,
        "per_class_f1": report["per_class_f1"] if report else None,
    }


def run(
    dataset: str, rungs: list[str] | None = None, seeds: list[int] | None = None,
    nrows: int | None = None, epochs: int = 2, max_bins: int | None = None,
) -> dict:
    rungs = rungs or ALL_RUNGS
    seeds = seeds or [0]
    cfg = load_config(dataset=dataset)
    processed_dir = resolved_path(cfg, "processed_dir") / dataset

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    if nrows is not None:
        train_df = train_df.iloc[:nrows].reset_index(drop=True)
        val_df = val_df.iloc[: max(nrows // 4, 1)].reset_index(drop=True)
    train_df = train_df.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)

    class_names = sorted(train_df["canonical_label"].unique().tolist())

    results = []
    for rung in rungs:
        for seed in seeds:
            print(f"[10] Rung {rung}, seed {seed} ...")
            r = run_one_rung(rung, dataset, seed, train_df, val_df, class_names, epochs, max_bins)
            print(f"[10]   F_e={r['f_e']}, val_macro_f1={r['best_val_macro_f1']:.4f}, test_macro_f1={r['macro_f1']}")
            results.append(r)

    out_dir = Path(__file__).parents[1] / cfg.run.out_dir / f"{dataset}_temporal_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "temporal_ladder_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[10] Wrote {out_dir / 'temporal_ladder_results.json'}")
    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rungs", nargs="+", default=None, choices=ALL_RUNGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-bins", type=int, default=None)
    args = parser.parse_args()
    run(args.dataset, rungs=args.rungs, seeds=args.seeds, nrows=args.nrows, epochs=args.epochs, max_bins=args.max_bins)
