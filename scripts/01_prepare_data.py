"""01 — Prepare data: clean, canonicalise, subsample a raw NF-v3 CSV.

Chunked processing for laptops with limited RAM (~8 GB). Reads the CSV in
500K-row chunks, does cross-chunk dedup via a hash set, writes cleaned parquet,
then runs the stratified temporal subsample on the (smaller) cleaned parquet.

Laptop dev run (small slice):
    python scripts/01_prepare_data.py --dataset cicids2018 --nrows 200000

Full run (streaming, ~8 GB peak RAM):
    python scripts/01_prepare_data.py --dataset cicids2018

Override the default subsample target:
    python scripts/01_prepare_data.py --dataset cicids2018 --subsample-target 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from argus.config import load_config, resolved_path  # noqa: E402
from argus.constants import LABEL_COLS, RAW_FEATURE_COLS  # noqa: E402
from argus.data.canonical import canonicalise_labels  # noqa: E402
from argus.data.clean import clean, NEGATIVE_CLIP_COLS, MIN_VALID_MS, MAX_VALID_MS  # noqa: E402
from argus.data.loader import read_csv_chunks  # noqa: E402


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk cleaning: nulls, negatives, time range. NO cross-chunk dedup."""
    df = df.dropna(subset=["Attack", "FLOW_START_MILLISECONDS"])
    for col in NEGATIVE_CLIP_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    df["FLOW_START_MILLISECONDS"] = df["FLOW_START_MILLISECONDS"].astype("int64")
    df["FLOW_END_MILLISECONDS"] = df["FLOW_END_MILLISECONDS"].astype("int64")
    plausible = df["FLOW_START_MILLISECONDS"].between(MIN_VALID_MS, MAX_VALID_MS)
    df = df[plausible]
    return df.reset_index(drop=True)


def run(
    dataset: str,
    nrows: int | None = None,
    subsample_target: int | None = None,
    chunksize: int = 500_000,
) -> Path:
    cfg = load_config(dataset=dataset)
    csv_path = resolved_path(cfg, "dataset_dir") / f"NF-{_display_name(dataset)}-v3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    out_dir = resolved_path(cfg, "interim_dir") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_path = out_dir / "_cleaned_temp.parquet"

    target = subsample_target or cfg.data.subsample_target
    if nrows is not None:
        target = min(target, nrows)

    # ── Phase 1: chunked clean + cross-chunk dedup → temp parquet ──────────
    seen_hashes: set[int] = set()
    class_counts: dict[str, int] = defaultdict(int)
    total_read = 0
    total_deduped = 0
    total_kept = 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None

    print(f"[01] Phase 1: chunked clean + dedup (chunksize={chunksize}) ...")

    for chunk in read_csv_chunks(csv_path, chunksize=chunksize, nrows=nrows):
        before = len(chunk)
        chunk = _clean_chunk(chunk)
        chunk = canonicalise_labels(chunk, dataset)
        total_read += before

        # Cross-chunk dedup via 64-bit hashes of all feature columns
        dedup_cols = [c for c in RAW_FEATURE_COLS if c in chunk.columns]
        if dedup_cols:
            hashes = pd.util.hash_pandas_object(chunk[dedup_cols].astype(str), index=False)
            mask = ~hashes.isin(seen_hashes)
            seen_hashes.update(int(h) for h in hashes[mask].values)
            chunk = chunk[mask.values]
            total_deduped += before - len(chunk)

        if len(chunk) == 0:
            continue

        # Accumulate class counts for subsample planning
        for lbl in chunk["canonical_label"]:
            class_counts[lbl] += 1

        # Append to temp parquet via pyarrow ParquetWriter (streaming, low memory)
        table = pa.Table.from_pandas(chunk.reset_index(drop=True))
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema)
        writer.write_table(table)

        total_kept += len(chunk)

        # Progress
        seen_mb = len(seen_hashes) * 72 / (1024 * 1024)
        print(f"[01]   Read {total_read:,} rows, kept {total_kept:,} "
              f"(deduped {total_deduped:,}), hash set ~{seen_mb:.0f} MB, "
              f"{len(class_counts)} classes seen")

    if writer is not None:
        writer.close()

    print(f"[01] Phase 1 done: {total_read:,} read → {total_kept:,} kept "
          f"({total_deduped:,} cross-chunk duplicates removed)")
    print(f"[01] Classes found: {len(class_counts)}")

    # ── Phase 2: streaming subsample via per-class quotas ──────────────────
    print(f"[01] Phase 2: streaming subsample (target: {target:,}) ...")

    # Compute per-class quotas from accumulated Phase 1 counts
    minority_threshold = cfg.data.minority_threshold
    benign_floor_fraction = cfg.data.benign_floor_fraction

    # Minority classes: keep all
    minority_classes = {c for c, n in class_counts.items() if n <= minority_threshold}
    minority_quota = sum(class_counts[c] for c in minority_classes)

    remaining = max(target - minority_quota, 0)
    benign_floor = int(benign_floor_fraction * target)
    non_benign_majority = [c for c in class_counts if c not in minority_classes and c != "benign"]

    # Benign quota (floored)
    benign_count = class_counts.get("benign", 0)
    benign_quota = min(benign_count, max(benign_floor, remaining // 2))
    remaining -= benign_quota

    # Distribute remaining among non-benign majority classes proportionally
    total_non_benign = sum(class_counts[c] for c in non_benign_majority)
    quotas: dict[str, int] = {}
    for c in minority_classes:
        quotas[c] = class_counts[c]  # keep all
    quotas["benign"] = benign_quota
    for c in non_benign_majority:
        if total_non_benign > 0 and remaining > 0:
            share = int(remaining * class_counts[c] / total_non_benign)
            quotas[c] = min(class_counts[c], share)
        else:
            quotas[c] = 0

    print(f"[01] Quotas: {len(quotas)} classes")
    for c, q in sorted(quotas.items()):
        print(f"[01]   {c}: {class_counts[c]:,} → {q:,}")

    # Streaming subsample: read temp parquet in chunks, keep rows by class ratio
    kept_frames: list[pd.DataFrame] = []
    counters: dict[str, int] = defaultdict(int)

    for batch in pq.ParquetFile(temp_path).iter_batches(batch_size=chunksize):
        chunk = batch.to_pandas()
        mask = pd.Series(False, index=chunk.index)
        for cls_name, quota in quotas.items():
            if quota == 0:
                continue
            cls_mask = chunk["canonical_label"] == cls_name
            cls_rows = chunk[cls_mask]
            n_available = len(cls_rows)
            n_needed = quota - counters[cls_name]
            if n_needed <= 0:
                continue
            if n_needed >= n_available:
                mask[cls_mask] = True
                counters[cls_name] += n_available
            else:
                # Take first n_needed (rows are time-ordered within class)
                cls_indices = cls_rows.index[:n_needed]
                mask[cls_indices] = True
                counters[cls_name] += n_needed
        if mask.any():
            kept_frames.append(chunk[mask])

    df = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    pre_counts = dict(class_counts)
    post_counts = {c: counters[c] for c in quotas}

    # Write final output
    out_path = out_dir / "cleaned.parquet"
    df.to_parquet(out_path, index=False)

    # Clean up temp file (best-effort; Windows may hold the handle briefly)
    try:
        if temp_path.exists():
            temp_path.unlink()
    except (PermissionError, OSError):
        pass

    # Save report
    clean_report = {
        "input_rows": total_read,
        "dropped_null": 0,
        "dropped_duplicates": total_deduped,
        "dropped_implausible_time": 0,
        "output_rows": total_kept,
    }
    subsample_report = {
        "target_total": target,
        "realised_total": len(df),
        "pre_counts": pre_counts,
        "post_counts": post_counts,
    }
    report = {
        "clean": clean_report,
        "subsample": subsample_report,
        "class_counts_from_chunks": dict(class_counts),
    }
    with open(out_dir / "subsample_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[01] Wrote {out_path} ({out_path.stat().st_size / 1024**2:.0f} MB)")
    return out_path


def _display_name(dataset: str) -> str:
    return {
        "cicids2018": "CICIDS2018",
        "ton_iot": "ToN-IoT",
        "unsw_nb15": "UNSW-NB15",
        "bot_iot": "BoT-IoT",
    }[dataset]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--subsample-target", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    run(args.dataset, nrows=args.nrows, subsample_target=args.subsample_target,
        chunksize=args.chunksize)
