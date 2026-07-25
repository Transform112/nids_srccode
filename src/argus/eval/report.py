"""Assemble a complete run report from all metric families.

Writes ``metrics.json`` in the run directory with every number that traces to
a ``run_id``. See docs/08_EVALUATION.md and docs/12_IMPLEMENTATION_PLAN.md §4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def assemble_run_report(
    run_dir: str | Path,
    run_id: str,
    dataset: str,
    protocol: str,
    model_name: str,
    closed_set: dict[str, Any] | None = None,
    open_set: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    selective: dict[str, Any] | None = None,
    few_shot: dict[str, Any] | None = None,
    adversarial: dict[str, Any] | None = None,
    temporal: dict[str, Any] | None = None,
    deployment: dict[str, Any] | None = None,
    xai: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect all metric families into a single report dict.

    Only non-None sections are included. Every reported number traces to this
    ``run_id``, satisfying standing rule 8.
    """
    report: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset,
        "protocol": protocol,
        "model": model_name,
    }
    if closed_set is not None:
        report["closed_set"] = closed_set
    if open_set is not None:
        report["open_set"] = open_set
    if calibration is not None:
        report["calibration"] = calibration
    if selective is not None:
        report["selective"] = selective
    if few_shot is not None:
        report["few_shot"] = few_shot
    if adversarial is not None:
        report["adversarial"] = adversarial
    if temporal is not None:
        report["temporal"] = temporal
    if deployment is not None:
        report["deployment"] = deployment
    if xai is not None:
        report["xai"] = xai
    if extra is not None:
        report.update(extra)
    return report


def save_run_report(report: dict[str, Any], run_dir: str | Path) -> Path:
    """Write ``metrics.json`` to the run directory. Returns the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    return path


def aggregate_over_seeds(
    seed_reports: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Aggregate per-seed reports into mean ± std.

    Args:
        seed_reports: one report dict per seed.
        metric_keys: dot-separated paths to scalar metrics, e.g.
            ``["closed_set.macro_f1", "calibration.ece"]``.

    Returns:
        Dict mapping each key to ``{"mean": ..., "std": ...}``.
    """
    result: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [_get_nested(r, key) for r in seed_reports]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            result[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        else:
            result[key] = {"mean": float("nan"), "std": float("nan")}
    return result


def _get_nested(d: dict[str, Any], key_path: str) -> Any:
    """Get a nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
