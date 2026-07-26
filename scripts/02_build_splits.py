"""02 — Build Protocol A/B/B2 splits and run the leakage audit.

Usage:
    python scripts/02_build_splits.py --dataset cicids2018 --protocol A
    python scripts/02_build_splits.py --dataset cicids2018 --protocol B --holdout-index 0

Protocol A writes to data/processed/<dataset>/ directly. Protocol B writes to
data/processed/<dataset>/holdout_b<i>/ (one subdirectory per holdout set) so
the 5 open-set training runs never clobber the closed-set pipeline or each
other, and persists the full holdout enumeration to holdout_sets.json — the
single source of truth every later script (03/04/05/07/08) reads instead of
re-sampling (re-sampling is order-sensitive in its input list and silently
produced different sets in different scripts).
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
from argus.utils.io import holdout_subdir  # noqa: E402


def write_holdout_sets(cfg, df, processed_root: Path) -> list[list[str]]:
    """Sample and persist the canonical holdout enumeration (holdout_sets.json).

    Attack classes are SORTED before sampling: `df.unique()` order is
    first-seen order, which differs between scripts reading raw splits and
    scripts reading the alphabetised class vocab — same seed, different
    input order, different holdout sets. Sorting makes the sample
    reproducible everywhere.
    """
    attack_classes = sorted(str(c) for c in df["canonical_label"].unique() if c != "benign")
    holdouts = [
        [str(c) for c in h]
        for h in sample_holdout_sets(
            attack_classes,
            holdout_size=min(cfg.data.holdout_size, len(attack_classes)),
            repeats=cfg.data.holdout_repeats,
            seed=cfg.data.holdout_seed,
        )
    ]
    path = processed_root / "holdout_sets.json"
    processed_root.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(holdouts, f, indent=2)
    print(f"[02] Wrote {len(holdouts)} holdout sets to {path}")
    return holdouts


def run(dataset: str, protocol: str = "A", holdout_index: int | None = None) -> Path:
    cfg = load_config(dataset=dataset)
    interim_path = resolved_path(cfg, "interim_dir") / dataset / "cleaned.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(f"Run 01_prepare_data.py first: {interim_path} not found")

    import pandas as pd
    df = pd.read_parquet(interim_path)

    processed_root = resolved_path(cfg, "processed_dir") / dataset

    if protocol == "A":
        out_dir = processed_root
        splits = protocol_a_split(df, label_col="canonical_label")
        holdout_classes = None
    elif protocol == "B":
        holdouts = write_holdout_sets(cfg, df, processed_root)
        idx = holdout_index if holdout_index is not None else 0
        if not 0 <= idx < len(holdouts):
            raise ValueError(f"--holdout-index {idx} out of range (0..{len(holdouts) - 1})")
        holdout_classes = holdouts[idx]
        out_dir = holdout_subdir(processed_root, idx)
        splits = protocol_b_split(df, holdout_classes=holdout_classes, label_col="canonical_label")
        print(f"[02] Protocol B holdout set {idx}: {holdout_classes}")
        print(f"[02] Output dir: {out_dir} (train/val exclude these classes entirely; "
              f"test contains them relabelled UNKNOWN)")
    else:
        raise ValueError(f"Protocol {protocol} not supported by this script yet")

    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit_split(splits, protocol=protocol, label_col="canonical_label")
    identity_report = identity_leakage_audit(
        splits["train"], splits["test"], label_col="canonical_label"
    )
    report["identity_leakage"] = identity_report
    if holdout_classes is not None:
        report["holdout_index"] = holdout_index if holdout_index is not None else 0
        report["holdout_classes"] = holdout_classes

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
    parser.add_argument("--holdout-index", type=int, default=None,
                        help="Protocol B: which holdout set (0..repeats-1) to build splits for")
    args = parser.parse_args()
    run(args.dataset, protocol=args.protocol, holdout_index=args.holdout_index)
