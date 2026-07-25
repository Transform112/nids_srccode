"""Latency instrumentation for deployment benchmarking.

See docs/08_EVALUATION.md §5.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

import numpy as np


class Timer:
    """Wall-clock stopwatch for per-flow or per-batch latency measurement."""

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) * 1000.0

    def reset(self) -> float:
        """Return elapsed ms and restart."""
        t = self.elapsed_ms()
        self._start = time.perf_counter()
        return t


def latency_percentiles(
    measurements: list[float],
    percentiles: tuple[int, ...] = (50, 90, 95, 99),
) -> dict[str, float]:
    """Compute latency percentiles from a list of millisecond measurements.

    Args:
        measurements: List of per-sample latencies in ms.
        percentiles: Which percentiles to report.

    Returns:
        Dict with keys like ``p50_ms``, ``p99_ms``, plus ``mean_ms`` and ``max_ms``.
    """
    if not measurements:
        return {"mean_ms": 0.0, "max_ms": 0.0, **{f"p{p}_ms": 0.0 for p in percentiles}}
    arr = np.asarray(measurements, dtype=np.float64)
    result = {"mean_ms": float(np.mean(arr)), "max_ms": float(np.max(arr))}
    for p in percentiles:
        result[f"p{p}_ms"] = float(np.percentile(arr, p))
    return result


class LatencyTracker:
    """Accumulates per-sample latency measurements with bounded memory."""

    def __init__(self, max_samples: int = 100_000) -> None:
        self._buffer: Deque[float] = deque(maxlen=max_samples)

    def record(self, latency_ms: float) -> None:
        self._buffer.append(latency_ms)

    def record_batch(self, latencies_ms: list[float]) -> None:
        self._buffer.extend(latencies_ms)

    def summarize(self, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
        return latency_percentiles(list(self._buffer), percentiles=percentiles)

    def count(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
