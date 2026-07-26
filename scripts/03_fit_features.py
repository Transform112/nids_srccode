"""03 — Fit the feature pipeline on train, transform all splits, save artifacts.

Usage:
    python scripts/03_fit_features.py --dataset cicids2018
    python scripts/03_fit_features.py --dataset cicids2018 --holdout-index 0

With --holdout-index the pipeline is fit on the Protocol-B holdout-excluded
train split (data/processed/<dataset>/holdout_b<i>/) and its artifacts land in
data/artifacts/<dataset>/holdout_b<i>/ — the pipeline must be refit per
holdout because its train split differs (fit-on-train-only, standing rule).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from argus.config import load_config, resolved_path  # noqa: E402
from argus.features.pipeline import FeaturePipeline  # noqa: E402
from argus.utils.io import holdout_subdir  # noqa: E402


def run(dataset: str, holdout_index: int | None = None) -> Path:
    cfg = load_config(dataset=dataset)
    processed_dir = holdout_subdir(resolved_path(cfg, "processed_dir") / dataset, holdout_index)
    artifact_dir = holdout_subdir(resolved_path(cfg, "artifact_dir") / dataset, holdout_index)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    train = pd.read_parquet(processed_dir / "train.parquet")

    pipeline = FeaturePipeline(
        protocol_topk=cfg.features.protocol_topk,
        l7_proto_topk=cfg.features.l7_proto_topk,
        dst_port_topk=cfg.features.dst_port_topk,
        quantile_n=cfg.features.quantile_n,
        quantile_subsample=cfg.features.quantile_subsample,
        clip_post_transform=cfg.features.clip_post_transform,
        te1_enabled=cfg.features.te1_enabled,
        te2_enabled=cfg.features.te2_enabled,
    )
    print("[03] Fitting feature pipeline on train split ...")
    pipeline.fit(train)
    a_idx, b_idx = pipeline.assert_channels()
    print(f"[03] F_e = {len(pipeline.feature_names_)} (channel A: {len(a_idx)}, channel B: {len(b_idx)})")

    pipeline_path = artifact_dir / "feature_pipeline.joblib"
    pipeline.save(pipeline_path)
    print(f"[03] Saved pipeline to {pipeline_path}")

    for split_name in ("train", "val", "test"):
        split_path = processed_dir / f"{split_name}.parquet"
        if not split_path.exists():
            continue
        df = pd.read_parquet(split_path)
        transformed = pipeline.transform(df)
        transformed["canonical_label"] = df["canonical_label"].values
        if "label_pre_holdout" in df.columns:
            transformed["label_pre_holdout"] = df["label_pre_holdout"].values
        transformed["FLOW_START_MILLISECONDS"] = df["FLOW_START_MILLISECONDS"].values
        transformed["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].values
        transformed["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].values
        transformed["L4_SRC_PORT"] = df["L4_SRC_PORT"].values
        transformed["L4_DST_PORT"] = df["L4_DST_PORT"].values
        out_path = processed_dir / f"{split_name}_features.parquet"
        transformed.to_parquet(out_path, index=False)
        print(f"[03] Wrote {out_path} ({len(transformed)} rows, {len(pipeline.feature_names_)} features)")

    return pipeline_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Protocol B: fit on holdout_b<i>'s train split instead")
    args = parser.parse_args()
    run(args.dataset, holdout_index=args.holdout_index)
