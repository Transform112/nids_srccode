"""02 — Build Protocol A/B/B2 splits and run the leakage audit.

Usage:
    python scripts/02_build_splits.py --dataset cicids2018 --protocol A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from argus.config import load_config, resolved_path  # noqa: E402
from argus.data.audit import identity_leakage_audit  # noqa: E402
from argus.data.audit import audit_split  # noqa: E402
from argus.data.splits import protocol_a_split, protocol_b_split, sample_holdout_sets  # noqa: E402


def run(dataset: str, protocol: str = "A") -> Path:
    cfg = load_config(dataset=dataset)
    interim_path = resolved_path(cfg, "interim_dir") / dataset / "cleaned.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(f"Run 01_prepare_data.py first: {interim_path} not found")

    import pandas as pd
    df = pd.read_parquet(interim_path)

    out_dir = resolved_path(cfg, "processed_dir") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if protocol == "A":
        splits = protocol_a_split(df, label_col="canonical_label")
    elif protocol == "B":
        attack_classes = [c for c in df["canonical_label"].unique() if c != "benign"]
        holdouts = sample_holdout_sets(
            attack_classes,
            holdout_size=cfg.data.holdout_size,
            repeats=cfg.data.holdout_repeats,
            seed=cfg.data.holdout_seed,
        )
        splits = protocol_b_split(df, holdout_classes=holdouts[0], label_col="canonical_label")
        print(f"[02] Protocol B holdout classes: {holdouts[0]}")
    else:
        raise ValueError(f"Protocol {protocol} not supported by this script yet")

    report = audit_split(splits, protocol=protocol, label_col="canonical_label")
    identity_report = identity_leakage_audit(
        splits["train"], splits["test"], label_col="canonical_label"
    )
    report["identity_leakage"] = identity_report

    for name in ("train", "val", "test"):
        splits[name].to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"[02] Wrote {out_dir / f'{name}.parquet'} ({len(splits[name])} rows)")

    with open(out_dir / "split_audit.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[02] Identity floor macro-F1: {identity_report['identity_floor_macro_f1']:.4f}")
    print(f"[02] Unseen (src,dst) pair rate: {identity_report['unseen_pair_rate']:.4f}")

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", default="A", choices=["A", "B", "B2"])
    args = parser.parse_args()
    run(args.dataset, protocol=args.protocol)
