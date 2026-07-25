"""15 — Regenerate paper tables/CSVs from whatever result JSON files exist
under `results/runs/` for a dataset (baselines, adversarial, xai, deployment).

This is intentionally a thin aggregator, not a new results format: it reads
the JSON each earlier script already writes and reshapes it into flat CSVs
under `results/tables/`, one per source, so every number in a CSV traces back
to the JSON (and therefore the run) it came from (docs/TODO.md standing rule 11).

Usage:
    python scripts/15_make_tables.py --dataset cicids2018
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from argus.config import load_config  # noqa: E402


def _flatten_baselines(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    rows = []
    for name, report in data.items():
        if "macro_f1" not in report:
            continue
        row = {"baseline": name, "macro_f1": report["macro_f1"]}
        for cls, f1 in report.get("per_class_f1", {}).items():
            row[f"f1_{cls}"] = f1
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _flatten_adversarial(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {name: pd.DataFrame(rows) for name, rows in data.items() if rows}


def _flatten_xai(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for key in ("native_attribution", "explainer_metrics", "triage_reports"):
        if data.get(key):
            out[key] = pd.DataFrame(data[key])
    if data.get("unknown_cluster_validation"):
        out["unknown_cluster_validation"] = pd.DataFrame([data["unknown_cluster_validation"]])
    return out


def _flatten_deployment(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    rows = [{"model": name, **report} for name, report in data.items()]
    return pd.DataFrame(rows) if rows else None


def run(dataset: str) -> dict[str, Path]:
    cfg = load_config(dataset=dataset)
    run_root = Path(__file__).parents[1] / cfg.run.out_dir
    tables_dir = Path(__file__).parents[1] / "results" / "tables" / dataset
    tables_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    baselines_df = _flatten_baselines(run_root / f"{dataset}_baselines" / "baseline_results.json")
    if baselines_df is not None:
        out = tables_dir / "T_baselines.csv"
        baselines_df.to_csv(out, index=False)
        written["baselines"] = out

    for name, df in _flatten_adversarial(run_root / f"{dataset}_adversarial" / "adversarial_results.json").items():
        out = tables_dir / f"T_adversarial_{name}.csv"
        df.to_csv(out, index=False)
        written[f"adversarial_{name}"] = out

    for name, df in _flatten_xai(run_root / f"{dataset}_xai" / "xai_results.json").items():
        out = tables_dir / f"T_xai_{name}.csv"
        df.to_csv(out, index=False)
        written[f"xai_{name}"] = out

    deployment_df = _flatten_deployment(run_root / f"{dataset}_deployment" / "deployment_report.json")
    if deployment_df is not None:
        out = tables_dir / "T_deployment.csv"
        deployment_df.to_csv(out, index=False)
        written["deployment"] = out

    print(f"[15] Wrote {len(written)} table(s) to {tables_dir}:")
    for name, path in written.items():
        print(f"[15]   {name} -> {path}")
    if not written:
        print("[15] No result JSON files found yet — run scripts 11-14 first.")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run(args.dataset)
