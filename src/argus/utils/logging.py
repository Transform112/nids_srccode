"""Structured run logging.

Writes one JSON line per event to ``logs/events.jsonl`` inside the run directory.
Supplements the tqdm/rich progress bars with machine-readable provenance.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Append-only JSONL event log for a single run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "logs" / "events.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

    def log(self, event: str, **fields: Any) -> None:
        """Append one structured event."""
        rec = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 3),
        }
        rec.update(fields)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def log_metric(self, step: int, **metrics: float) -> None:
        """Log a metric at a given step/epoch."""
        self.log("metric", step=step, **metrics)

    def log_config(self, cfg: Any) -> None:
        """Record the resolved config as a single event.

        Converts the OmegaConf DictConfig to a plain dict for serialisation.
        """
        from omegaconf import OmegaConf

        as_dict = OmegaConf.to_container(cfg, resolve=True)
        self.log("config", config=as_dict)

    def log_env(self, extra: dict[str, Any] | None = None) -> None:
        """Record git SHA, pip freeze, and platform info."""
        import platform
        import sys

        import torch

        info: dict[str, Any] = {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        }
        try:
            import torch_geometric
            info["torch_geometric"] = torch_geometric.__version__
        except ImportError:
            pass
        try:
            import numpy as np
            info["numpy"] = np.__version__
        except ImportError:
            pass
        if extra:
            info.update(extra)
        self.log("env", **info)
