"""Run registry backed by a JSONL file.

Every completed run appends one line to ``results/runs/registry.jsonl``.
Experiment runners skip runs already in the registry, making every script
resumable across Kaggle sessions.

See docs/12_IMPLEMENTATION_PLAN.md §4.5.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunRegistry:
    """Append-only JSONL registry of completed runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._ids.add(rec["run_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    @property
    def completed_ids(self) -> set[str]:
        return self._ids

    def is_complete(self, run_id: str) -> bool:
        return run_id in self._ids

    def register(
        self,
        run_id: str,
        dataset: str,
        protocol: str,
        model: str,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a completed run to the registry.

        Idempotent: if *run_id* already exists the record is not duplicated.
        """
        if self.is_complete(run_id):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "run_id": run_id,
            "dataset": dataset,
            "protocol": protocol,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
        }
        if metrics is not None:
            rec["metrics"] = metrics
        if extra is not None:
            rec.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self._ids.add(run_id)


def register_run(
    registry_path: str | Path,
    run_id: str,
    dataset: str,
    protocol: str,
    model: str,
    metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Convenience: register a single run and return."""
    reg = RunRegistry(registry_path)
    reg.register(run_id, dataset, protocol, model, metrics=metrics, extra=extra or None)


def is_run_complete(registry_path: str | Path, run_id: str) -> bool:
    """Check whether a run has already been completed."""
    reg = RunRegistry(registry_path)
    return reg.is_complete(run_id)
